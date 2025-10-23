"""
Results management and reporting.

Handles:
- Saving results with taxonomy structure
- Generating comparison reports
- Computing metrics (ASR, precision, etc.)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class ResultsManager:
    """Manages benchmark results and generates reports."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        prompts: List,
        taxonomy: Optional[str] = None,
        taxonomy_structure: Optional[Dict] = None,
    ):
        """
        Save benchmark results to JSON.

        Args:
            all_results: Dict mapping detector names to result lists
            prompts: List of ProbePrompt objects
            taxonomy: Taxonomy used (if any)
            taxonomy_structure: Taxonomy structure (if any)
        """
        # Main results file
        results_file = self.output_dir / "results.json"

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_prompts": len(prompts),
                "detectors": list(all_results.keys()),
                "taxonomy": taxonomy,
            },
            "results": all_results,
        }

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        # Save taxonomy-structured results if available
        if taxonomy and taxonomy_structure:
            self._save_taxonomy_results(all_results, taxonomy_structure, taxonomy)

    def _save_taxonomy_results(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        taxonomy_structure: Dict,
        taxonomy: str,
    ):
        """
        Save results organized by taxonomy structure.

        Creates directory structure mirroring the taxonomy.
        Example: results/owasp/llm01/detector_results.json
        """
        taxonomy_dir = self.output_dir / taxonomy
        taxonomy_dir.mkdir(parents=True, exist_ok=True)

        # For each category in taxonomy
        for category, category_prompts in taxonomy_structure.items():
            # Extract just the category suffix (e.g., "llm01" from "owasp:llm01")
            category_suffix = category.split(":")[-1] if ":" in category else category
            category_dir = taxonomy_dir / category_suffix
            category_dir.mkdir(parents=True, exist_ok=True)

            # Collect results for this category across all detectors
            category_results = {}

            for detector_name, detector_results in all_results.items():
                # Filter results for prompts in this category
                prompt_texts = {p.text for p in category_prompts}
                filtered = [r for r in detector_results if r["prompt"] in prompt_texts]
                category_results[detector_name] = filtered

            # Save category results
            category_file = category_dir / "results.json"
            with open(category_file, "w") as f:
                json.dump(
                    {
                        "category": category,
                        "total_prompts": len(category_prompts),
                        "results": category_results,
                    },
                    f,
                    indent=2,
                )

    def generate_report(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        prompts: List,
        taxonomy: Optional[str] = None,
    ):
        """
        Generate a markdown comparison report.

        Args:
            all_results: Dict mapping detector names to result lists
            prompts: List of ProbePrompt objects
            taxonomy: Taxonomy used (if any)
        """
        report_file = self.output_dir / "report.md"

        with open(report_file, "w") as f:
            # Header
            f.write("# Guardrails Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Prompts:** {len(prompts)}\n\n")
            f.write(f"**Detectors:** {', '.join(all_results.keys())}\n\n")
            if taxonomy:
                f.write(f"**Taxonomy:** {taxonomy}\n\n")

            f.write("---\n\n")

            # Prompt breakdown per detector
            f.write("## Prompt Breakdown\n\n")
            f.write("| Detector | Total Prompts | Adversarial | Benign |\n")
            f.write("|----------|---------------|-------------|--------|\n")

            for detector_name, results in all_results.items():
                total = len(results)
                benign_count = sum(1 for r in results if r.get("is_benign", False))
                adversarial_count = total - benign_count
                f.write(
                    f"| {detector_name} | {total} | {adversarial_count} | {benign_count} |\n"
                )

            f.write("\n")

            # Latency statistics per detector
            f.write("## Latency Statistics\n\n")
            f.write("| Detector | Mean | Median (P50) | P95 | P99 | Min | Max |\n")
            f.write("|----------|------|--------------|-----|-----|-----|-----|\n")

            for detector_name, results in all_results.items():
                latency_stats = self._calculate_latency_stats(results)
                f.write(
                    f"| {detector_name} | {latency_stats['mean']:.1f}ms | "
                    f"{latency_stats['p50']:.1f}ms | {latency_stats['p95']:.1f}ms | "
                    f"{latency_stats['p99']:.1f}ms | {latency_stats['min']:.1f}ms | "
                    f"{latency_stats['max']:.1f}ms |\n"
                )

            f.write("\n")

            # Per-probe results with F1/Precision/Recall
            f.write("## Results by Probe\n\n")

            # Group results by probe_name for each detector
            for detector_name, results in all_results.items():
                f.write(f"### {detector_name}\n\n")

                by_probe = self._group_by_probe(results)

                f.write(
                    "| Probe | Type | Total | TP | TN | FP | FN | Precision | Recall | F1 Score |\n"
                )
                f.write(
                    "|-------|------|-------|----|----|----|----|-----------|--------|----------|\n"
                )

                # Build a mapping of base probe names to their adversarial/benign results
                probe_groups = {}
                for probe_name, probe_results in by_probe.items():
                    if probe_name.startswith("benign."):
                        # Benign probe - extract base name
                        base_name = probe_name.replace("benign.", "")
                        if base_name not in probe_groups:
                            probe_groups[base_name] = {"adversarial": [], "benign": []}
                        probe_groups[base_name]["benign"] = probe_results
                    else:
                        # Adversarial probe
                        if probe_name not in probe_groups:
                            probe_groups[probe_name] = {"adversarial": [], "benign": []}
                        probe_groups[probe_name]["adversarial"] = probe_results

                # Now output in organized groups
                for base_probe_name in sorted(probe_groups.keys()):
                    group = probe_groups[base_probe_name]
                    adv_results = group["adversarial"]
                    benign_results = group["benign"]

                    # Show adversarial metrics first
                    if adv_results:
                        adv_metrics = self._calculate_classification_metrics(
                            adv_results, is_benign=False
                        )
                        f.write(
                            f"| {base_probe_name} | Adversarial | {adv_metrics['total']} | "
                            f"{adv_metrics['tp']} | {adv_metrics['tn']} | "
                            f"{adv_metrics['fp']} | {adv_metrics['fn']} | "
                            f"{adv_metrics['precision']:.2%} | {adv_metrics['recall']:.2%} | "
                            f"{adv_metrics['f1']:.3f} |\n"
                        )

                    # Show benign metrics second
                    if benign_results:
                        benign_metrics = self._calculate_classification_metrics(
                            benign_results, is_benign=True
                        )
                        f.write(
                            f"| {base_probe_name} | Benign | {benign_metrics['total']} | "
                            f"{benign_metrics['tp']} | {benign_metrics['tn']} | "
                            f"{benign_metrics['fp']} | {benign_metrics['fn']} | "
                            f"{benign_metrics['precision']:.2%} | {benign_metrics['recall']:.2%} | "
                            f"{benign_metrics['f1']:.3f} |\n"
                        )

                    # Show combined metrics third (if we have both)
                    if adv_results and benign_results:
                        combined_metrics = self._calculate_combined_metrics(
                            adv_results, benign_results
                        )
                        f.write(
                            f"| {base_probe_name} | **Combined** | {combined_metrics['total']} | "
                            f"{combined_metrics['tp']} | {combined_metrics['tn']} | "
                            f"{combined_metrics['fp']} | {combined_metrics['fn']} | "
                            f"**{combined_metrics['precision']:.2%}** | **{combined_metrics['recall']:.2%}** | "
                            f"**{combined_metrics['f1']:.3f}** |\n"
                        )

                f.write("\n")

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for a set of results."""
        valid_results = [r for r in results if "error" not in r or r["error"] is None]
        total = len(valid_results)

        if total == 0:
            return {
                "total_tests": len(results),
                "detection_rate": 0.0,
                "avg_score": 0.0,
                "errors": len(results),
            }

        detected_count = sum(1 for r in valid_results if r.get("detected", False))
        total_score = sum(r.get("score", 0.0) for r in valid_results)

        return {
            "total_tests": total,
            "detection_rate": detected_count / total if total > 0 else 0.0,
            "avg_score": total_score / total if total > 0 else 0.0,
            "errors": len(results) - total,
        }

    def _group_by_probe(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by probe name."""
        by_probe = defaultdict(list)
        for result in results:
            probe_name = result.get("probe_name", "unknown")
            by_probe[probe_name].append(result)
        return dict(by_probe)

    def _calculate_latency_stats(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate latency statistics from results."""
        latencies = [
            r.get("latency_ms", 0.0)
            for r in results
            if "error" not in r or r["error"] is None
        ]

        if not latencies:
            return {
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        def percentile(p):
            """Calculate percentile from sorted list."""
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            if f == c:
                return latencies_sorted[f]
            return latencies_sorted[f] * (c - k) + latencies_sorted[c] * (k - f)

        return {
            "mean": sum(latencies) / n,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "min": latencies_sorted[0],
            "max": latencies_sorted[-1],
        }

    def _calculate_classification_metrics(
        self, results: List[Dict[str, Any]], is_benign: bool
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics (TP, TN, FP, FN, Precision, Recall, F1).

        For adversarial prompts: TP=detected, FN=not detected
        For benign prompts: TN=not detected, FP=detected
        """
        valid_results = [r for r in results if "error" not in r or r["error"] is None]
        total = len(valid_results)

        if total == 0:
            return {
                "total": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        detected = sum(1 for r in valid_results if r.get("detected", False))
        not_detected = total - detected

        if is_benign:
            # Benign prompts: TN = not detected (correct), FP = detected (incorrect)
            tp = 0
            tn = not_detected
            fp = detected
            fn = 0
        else:
            # Adversarial prompts: TP = detected (correct), FN = not detected (incorrect)
            tp = detected
            tn = 0
            fp = 0
            fn = not_detected

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _calculate_combined_metrics(
        self,
        adversarial_results: List[Dict[str, Any]],
        benign_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate combined classification metrics across both adversarial and benign prompts.

        TP = adversarial prompts detected
        TN = benign prompts not detected
        FP = benign prompts detected
        FN = adversarial prompts not detected
        """
        # Adversarial: detected = TP, not detected = FN
        adv_valid = [
            r for r in adversarial_results if "error" not in r or r["error"] is None
        ]
        tp = sum(1 for r in adv_valid if r.get("detected", False))
        fn = len(adv_valid) - tp

        # Benign: not detected = TN, detected = FP
        benign_valid = [
            r for r in benign_results if "error" not in r or r["error"] is None
        ]
        tn = sum(1 for r in benign_valid if not r.get("detected", False))
        fp = len(benign_valid) - tn

        total = len(adv_valid) + len(benign_valid)

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
