"""
Generic probe extraction from garak.

Supports multiple extraction modes:
- By probe spec (comma-separated list like garak)
- By tag filter (e.g., "owasp:llm01", "avid-effect")
- By module (e.g., "dan", "encoding", "lmrc")
- All probes

Extracts prompts while preserving taxonomy structure for reporting.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ProbePrompt:
    """A single prompt extracted from a probe."""

    text: str  # The actual prompt text
    probe_name: str  # Full probe name (e.g., "owasp.LLMTopTen")
    probe_class: str  # Probe class name
    module: str  # Probe module (e.g., "owasp", "dan")
    tags: List[str] = field(default_factory=list)  # Probe tags
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    is_benign: bool = False  # True if this is a benign (non-attack) prompt


class ProbeExtractor:
    """Extracts prompts from garak probes with multiple filtering options."""

    def __init__(self, verbose: bool = False, include_inactive: bool = False, deduplicate: bool = True):
        self.verbose = verbose
        self.include_inactive = include_inactive  # Include inactive "Full" probes
        self.deduplicate = deduplicate  # Remove duplicate prompts within each probe
        self.logger = logging.getLogger("probe_extractor")
        self._garak_loaded = False
        self._plugins = None
        self._config = None

    def _load_garak(self):
        """Lazy-load garak modules."""
        if self._garak_loaded:
            return

        try:
            from garak import _plugins, _config

            self._plugins = _plugins
            self._config = _config
            self._garak_loaded = True
            self.logger.info("Loaded garak modules")
        except ImportError as e:
            raise ImportError(
                f"Failed to import garak. Is garak installed and in PYTHONPATH? Error: {e}"
            )

    def list_all_probes(self) -> List[str]:
        """List all available garak probes."""
        self._load_garak()
        all_probes = self._plugins.enumerate_plugins("probes")
        # enumerate_plugins returns tuples of (name, active), extract just names
        probe_names = [name for name, active in all_probes]
        return sorted(probe_names)

    def parse_probe_spec(
        self, probe_spec: str, tag_filter: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Parse probe spec deterministically using garak's parse_plugin_spec.

        Args:
            probe_spec: Comma-separated probe names, "all", or specific probe.class
            tag_filter: Optional tag filter to apply (e.g., "owasp:llm01")

        Returns:
            Tuple of (selected_probes, rejected_probes) - both sorted for determinism
        """
        self._load_garak()

        spec = (
            probe_spec
            if probe_spec and probe_spec.lower() not in ("", "all")
            else "*"
        )

        # Use garak's parse_plugin_spec to get active probes efficiently
        selected, rejected = self._config.parse_plugin_spec(
            spec, category="probes", probe_tag_filter=tag_filter or ""
        )

        # If include_inactive, add inactive probes that match criteria
        if self.include_inactive:
            import importlib

            # Get all probes and filter for inactive ones
            all_probes = self._plugins.enumerate_plugins("probes")
            inactive_probes = [name for name, is_active in all_probes if not is_active]

            for probe_name in inactive_probes:
                # If tag filter specified, check if probe matches
                if tag_filter:
                    try:
                        # Import probe CLASS (not instance) to check tags
                        # Format: "probes.module.ClassName" -> "garak.probes.module"
                        clean_name = probe_name.replace("probes.", "", 1)
                        parts = clean_name.split(".")
                        module_path = f"garak.probes.{parts[0]}"
                        class_name = ".".join(parts[1:]) if len(parts) > 1 else parts[0]

                        probe_module = importlib.import_module(module_path)
                        probe_class = getattr(probe_module, class_name)

                        # Check class tags (no instantiation needed)
                        probe_tags = getattr(probe_class, 'tags', [])
                        has_matching_tag = any(tag.startswith(tag_filter) for tag in probe_tags)

                        if has_matching_tag:
                            selected.append(probe_name)
                            self.logger.info(f"Including inactive probe: {probe_name}")

                    except Exception as e:
                        self.logger.warning(f"Could not check tags for {probe_name}: {e}")
                else:
                    # No tag filter - include all inactive probes
                    selected.append(probe_name)
                    self.logger.info(f"Including inactive probe: {probe_name}")

        return sorted(selected), sorted(rejected)


    def extract_prompts(
        self,
        probe_spec: str = "all",
        tag_filter: Optional[str] = None,
        max_prompts_per_probe: Optional[int] = None,
    ) -> List[ProbePrompt]:
        """
        Extract prompts from probes based on spec and filters.

        Args:
            probe_spec: Probe specification (comma-separated names, "all", etc.)
            tag_filter: Optional tag filter (e.g., "owasp:llm01")
            max_prompts_per_probe: Optional limit on prompts per probe

        Returns:
            List of ProbePrompt objects
        """
        self._load_garak()

        # Parse probe spec to get list of probes
        selected_probes, rejected = self.parse_probe_spec(probe_spec, tag_filter)

        if rejected:
            self.logger.warning(f"Unknown probes: {rejected}")

        self.logger.info(f"Extracting prompts from {len(selected_probes)} probes")

        all_prompts = []
        for probe_name in selected_probes:
            try:
                prompts = self._extract_from_probe(probe_name, max_prompts_per_probe)
                all_prompts.extend(prompts)
                self.logger.info(f"Extracted {len(prompts)} prompts from {probe_name}")
            except Exception as e:
                self.logger.warning(f"Failed to extract from {probe_name}: {e}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()
                continue

        self.logger.info(f"Total prompts extracted: {len(all_prompts)}")
        return all_prompts

    def _extract_from_probe(
        self, probe_name: str, max_prompts: Optional[int] = None
    ) -> List[ProbePrompt]:
        """Extract prompts from a single probe."""
        # Save current random state, then set deterministic seed for this probe
        # This ensures each probe gets consistent random samples without affecting other probes
        import random
        import hashlib

        # Save the current random state
        saved_state = random.getstate()

        try:
            # Set seed based on probe name for deterministic sampling
            seed = int(hashlib.md5(probe_name.encode()).hexdigest()[:8], 16)
            random.seed(seed)

            plugin_name = (
                probe_name if probe_name.startswith("probes.") else f"probes.{probe_name}"
            )
            probe = self._plugins.load_plugin(plugin_name, config_root=self._config)

            probe_tags = getattr(probe, "tags", [])

            clean_name = probe_name.replace("probes.", "", 1)
            parts = clean_name.split(".")
            module = parts[0] if parts else "unknown"

            # Garak's load_plugin returns an instantiated probe object
            # Probes have 'prompts' (inputs to test) and optionally 'triggers' (expected outputs for detection)
            # We only extract prompts - triggers are detection targets, not test inputs
            prompt_texts = []
            if hasattr(probe, "prompts") and probe.prompts:
                try:
                    prompt_texts = list(probe.prompts)
                except (TypeError, StopIteration) as e:
                    self.logger.warning(f"Failed to iterate prompts for {probe_name}: {e}")
                    prompt_texts = []

            # Optionally deduplicate prompt texts while preserving order
            if self.deduplicate:
                seen = set()
                unique_texts = []
                for text in prompt_texts:
                    if isinstance(text, str) and text.strip() and text not in seen:
                        seen.add(text)
                        unique_texts.append(text)
            else:
                # No deduplication - just filter out non-strings and empty strings
                unique_texts = [text for text in prompt_texts if isinstance(text, str) and text.strip()]

            if max_prompts:
                unique_texts = unique_texts[:max_prompts]

            return [
                ProbePrompt(
                    text=text,
                    probe_name=probe_name,
                    probe_class=type(probe).__name__,
                    module=module,
                    tags=probe_tags,
                    metadata={},
                )
                for text in unique_texts
            ]
        finally:
            # Always restore the random state, even if extraction failed
            random.setstate(saved_state)

    def get_taxonomy_structure(
        self, prompts: List[ProbePrompt], taxonomy: Optional[str] = None
    ) -> Dict[str, List[ProbePrompt]]:
        """
        Organize prompts by taxonomy for structured reporting.

        Args:
            prompts: List of ProbePrompt objects
            taxonomy: Taxonomy to use (e.g., "owasp", "avid-effect"). If None, uses module.

        Returns:
            Dict mapping taxonomy categories to lists of prompts
        """
        structure = {}

        for prompt in prompts:
            if taxonomy:
                # Find matching tag
                category = None
                for tag in prompt.tags:
                    if tag.startswith(taxonomy):
                        category = tag
                        break
                if not category:
                    category = f"{taxonomy}:unknown"
            else:
                # Use module as category
                category = prompt.module

            if category not in structure:
                structure[category] = []
            structure[category].append(prompt)

        return structure

    def extract_benign_prompts(self, count: int = 100) -> List[ProbePrompt]:
        """
        Extract benign prompts from garak's base prompts (without attacks).

        Uses the base prompts from PromptInject framework - these are the legitimate
        tasks that attacks are injected into. By using them without attacks,
        we get realistic benign prompts that are semantically similar to attack prompts.

        Args:
            count: Number of benign prompts to generate

        Returns:
            List of ProbePrompt objects marked as benign
        """
        self._load_garak()

        benign_prompts = []

        try:
            from garak.resources.promptinject import prompt_data

            base_prompts = prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS
            sample_completions = [
                "I went to the store yesterday",
                "The cat sat on the mat",
                "Machine learning is a subset of artificial intelligence",
                "Python is a programming language",
                "Climate change affects global weather patterns",
                "The water cycle describes how water moves through Earth's systems",
                "Democracy is a system of government",
                "Photosynthesis is the process by which plants make food",
            ]

            prompt_idx = 0
            for base_prompt in base_prompts:
                if prompt_idx >= count:
                    break

                instruction = base_prompt.get("instruction", "")
                input_template = base_prompt.get("input", "{user_input}")
                label = base_prompt.get("label", "unknown")

                for completion in sample_completions:
                    if prompt_idx >= count:
                        break
                    full_prompt = instruction + input_template.replace(
                        "{user_input}", completion
                    )
                    benign_prompts.append(
                        ProbePrompt(
                            text=full_prompt,
                            probe_name="benign.promptinject_base",
                            probe_class=label,
                            module="benign",
                            tags=["benign", "promptinject_base"],
                            metadata={"source": "promptinject_base"},
                            is_benign=True,
                        )
                    )
                    prompt_idx += 1

        except ImportError as e:
            self.logger.warning(f"Could not import PromptInject data: {e}")
            simple_benign = [
                "What is the capital of France?",
                "How do I bake chocolate chip cookies?",
                "Explain photosynthesis in simple terms.",
                "What is the weather like today?",
                "Summarize the water cycle.",
                "How does a computer work?",
                "What are the benefits of exercise?",
                "Describe the solar system.",
            ]

            for i, text in enumerate(simple_benign[:count]):
                benign_prompts.append(
                    ProbePrompt(
                        text=text,
                        probe_name="benign.simple",
                        probe_class="simple",
                        module="benign",
                        tags=["benign", "simple"],
                        metadata={"source": "simple_fallback"},
                        is_benign=True,
                    )
                )
        self.logger.info(f"Extracted {len(benign_prompts)} benign prompts")
        return benign_prompts[:count]

    def generate_category_specific_benign_prompts(
        self,
        prompts: List[ProbePrompt],
        llm_config: Dict[str, Any],
    ) -> List[ProbePrompt]:
        """
        Generate category-specific benign prompts using an LLM.

        For each unique probe category (e.g., encoding.InjectBase64, promptinject.HijackKillHumans),
        generates the same number of benign prompts as there are adversarial prompts in that category.
        This ensures balanced testing where each category has equal adversarial and benign samples.

        Args:
            prompts: List of adversarial prompts to analyze for categories
            llm_config: LLM configuration dict with keys:
                - api_url: OpenAI-compatible API endpoint (e.g., vLLM with /v1)
                - model_name: Model name
                - api_key_env: Environment variable name containing API key
                - temperature: (optional) Temperature for generation
                - max_tokens: (optional) Max tokens per generation
                - timeout: (optional) Request timeout

        Returns:
            List of ProbePrompt objects marked as benign (same total count as input prompts)

        Raises:
            RuntimeError: If LLM API fails or configuration is invalid
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        # Validate config
        required_keys = ["api_url", "model_name", "api_key_env"]
        for key in required_keys:
            if key not in llm_config:
                raise RuntimeError(f"Missing required LLM config key: {key}")

        # Get API key from environment
        api_key_env = llm_config["api_key_env"]
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} not set. "
                f"Please set it with your LLM API key."
            )

        # Initialize OpenAI client (works with vLLM OpenAI-compatible endpoints)
        client = OpenAI(
            api_key=api_key,
            base_url=llm_config["api_url"],
            timeout=llm_config.get("timeout", 30),
        )
        categories = {}
        for prompt in prompts:
            category = prompt.probe_name
            if category not in categories:
                categories[category] = {
                    "probe_name": prompt.probe_name,
                    "probe_class": prompt.probe_class,
                    "module": prompt.module,
                    "tags": prompt.tags,
                    "examples": [],
                    "count": 0,  # Track how many adversarial prompts in this category
                }
            categories[category]["count"] += 1
            if len(categories[category]["examples"]) < 3:
                categories[category]["examples"].append(prompt.text[:200])
        total_prompts = sum(cat["count"] for cat in categories.values())
        self.logger.info(
            f"Generating {total_prompts} benign prompts across {len(categories)} probe categories"
        )
        benign_prompts = []
        generation_tasks = []

        for category_name, category_info in categories.items():
            category_count = category_info["count"]
            system_prompt = self._build_category_prompt_template(
                category_name, category_info, category_count
            )
            batch_size = 10
            for batch_start in range(0, category_count, batch_size):
                batch_count = min(batch_size, category_count - batch_start)
                batch_prompt = system_prompt.replace(
                    f"Generate {category_count}", f"Generate {batch_count}"
                )
                generation_tasks.append(
                    {
                        "category_name": category_name,
                        "category_info": category_info,
                        "batch_prompt": batch_prompt,
                        "batch_count": batch_count,
                    }
                )

        self.logger.info(
            f"Launching {len(generation_tasks)} parallel LLM generation tasks for {len(categories)} categories"
        )

        # Execute all generation tasks in parallel
        max_workers = min(20, len(generation_tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._generate_batch,
                    client,
                    task["batch_prompt"],
                    llm_config,
                ): task
                for task in generation_tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    generated_texts = future.result()
                    actual_count = len(generated_texts)
                    expected_count = task["batch_count"]

                    if actual_count < expected_count:
                        self.logger.warning(
                            f"LLM generated {actual_count} prompts but {expected_count} were requested "
                            f"for category {task['category_name']}"
                        )

                    for text in generated_texts[: task["batch_count"]]:
                        benign_prompts.append(
                            ProbePrompt(
                                text=text,
                                probe_name=f"benign.{task['category_name']}",
                                probe_class=task["category_info"]["probe_class"],
                                module="benign",
                                tags=["benign", "llm_generated"]
                                + task["category_info"]["tags"],
                                metadata={
                                    "source": "llm_generated",
                                    "source_category": task["category_name"],
                                    "generator_model": llm_config["model_name"],
                                },
                                is_benign=True,
                            )
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to generate benign prompts for category {task['category_name']}: {e}"
                    )

        expected_total = sum(cat["count"] for cat in categories.values())
        actual_total = len(benign_prompts)

        self.logger.info(
            f"Generated {actual_total} category-specific benign prompts (expected: {expected_total})"
        )

        if actual_total < expected_total:
            shortfall = expected_total - actual_total
            self.logger.warning(
                f"Generated {shortfall} fewer benign prompts than expected. "
                f"This may be due to LLM not generating requested number of prompts per batch."
            )
            print(
                f"⚠️  Warning: Generated {actual_total} benign prompts but expected {expected_total} "
                f"({shortfall} short). The LLM may not be generating enough prompts per request."
            )

        return benign_prompts

    def _generate_batch(
        self,
        client,
        batch_prompt: str,
        llm_config: Dict[str, Any],
    ) -> List[str]:
        """Generate a single batch of prompts via LLM (called in parallel)."""
        # Calculate max_tokens based on batch size (estimate ~80 tokens per prompt)
        # Extract count from prompt
        import re
        match = re.search(r'Generate (\d+)', batch_prompt)
        expected_count = int(match.group(1)) if match else 10
        calculated_max_tokens = max(500, expected_count * 80)

        response = client.chat.completions.create(
            model=llm_config["model_name"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates legitimate, harmless user queries. Generate EXACTLY the number of questions requested, one per line.",
                },
                {"role": "user", "content": batch_prompt},
            ],
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", calculated_max_tokens),
            n=1,
        )

        # Parse response
        generated_text = response.choices[0].message.content.strip()
        return self._parse_generated_prompts(generated_text)

    def _build_category_prompt_template(
        self, category_name: str, category_info: Dict[str, Any], count: int
    ) -> str:
        """Build LLM prompt template based on probe category."""

        # Extract module and class from category name
        # Format: "module.ClassName" or "probes.module.ClassName"
        clean_name = category_name.replace("probes.", "")
        parts = clean_name.split(".")
        module = parts[0] if parts else "unknown"
        probe_class = parts[1] if len(parts) > 1 else "unknown"

        # Category-specific templates
        templates = {
            "encoding": (
                f"Generate {count} legitimate, harmless questions about {probe_class.lower()} encoding "
                "that a helpful user might ask. These should be questions about understanding, "
                "using, or decoding this encoding format in normal contexts. "
                "Return ONLY the questions, one per line, without numbering or bullets."
            ),
            "promptinject": (
                f"Generate {count} legitimate, normal instruction-following questions "
                "that a user might ask an AI assistant. These should be straightforward tasks "
                "like summarization, translation, or information requests - NOT manipulation attempts. "
                "Return ONLY the questions, one per line, without numbering or bullets."
            ),
            "dan": (
                f"Generate {count} legitimate questions about creative writing, roleplay, "
                "or storytelling that don't attempt to bypass AI safety guidelines. "
                "Return ONLY the questions, one per line, without numbering or bullets."
            ),
            "jailbreak": (
                f"Generate {count} normal, straightforward questions that a user might ask "
                "an AI assistant for helpful information or assistance. "
                "Return ONLY the questions, one per line, without numbering or bullets."
            ),
            "malwaregen": (
                f"Generate {count} legitimate questions about software development, "
                "cybersecurity education, or system administration. "
                "Return ONLY the questions, one per line, without numbering or bullets."
            ),
        }

        # Get template for this module, or use generic template
        if module in templates:
            template = templates[module]
        else:
            # Generic template
            template = (
                f"Generate {count} legitimate, harmless questions related to the topic of "
                f"{probe_class.lower()} that a helpful user might ask. "
                "These should be normal, straightforward requests for information or assistance. "
                "Return ONLY the questions, one per line, without numbering or bullets."
            )

        # Add example context if available
        examples = category_info.get("examples", [])
        if examples:
            template += (
                f"\n\nFor context, here are examples of ATTACK prompts in this category "
                f"(generate OPPOSITE of these - legitimate questions instead):\n"
            )
            for i, example in enumerate(examples[:2], 1):
                template += f"{i}. {example}...\n"

        return template

    def _parse_generated_prompts(self, text: str) -> List[str]:
        """Parse LLM-generated text into individual prompts."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        prompts = []
        for line in lines:
            line = line.lstrip("0123456789.-*• ")
            if line:
                prompts.append(line)
        return prompts
