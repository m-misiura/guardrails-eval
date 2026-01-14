#!/usr/bin/env python3
"""
Guardrails Evaluation Framework

CLI tool to benchmark detector/guardrail effectiveness using garak's probe library.
Similar interface to garak, but for testing detectors instead of generators.
"""

import argparse
import sys
import datetime


def main(arguments=None):
    from src.command import (
        print_probes,
        print_detectors_config,
        start_logging,
        run_benchmark,
    )

    starttime = datetime.datetime.now()
    starttime_iso = starttime.isoformat()

    print(
        f"guardrails-eval: detector benchmarking using garak probes at {starttime_iso}"
    )

    parser = argparse.ArgumentParser(
        prog="python guardrails_eval.py",
        description="Benchmark guardrail detectors using garak's attack probe library",
        epilog="Uses garak probes to test detector effectiveness",
        allow_abbrev=False,
    )

    # SYSTEM OPTIONS
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="increase verbosity of output during runtime",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="output directory for results (default: results/)",
    )
    parser.add_argument(
        "--parallel_requests",
        type=int,
        default=1,
        help="how many detector requests to launch in parallel (default: 1)",
    )

    # DETECTOR CONFIGURATION
    parser.add_argument(
        "--detectors",
        "-d",
        type=str,
        required=False,
        help="path to detectors configuration YAML file",
    )

    # PROBE SELECTION (mirrors garak's interface)
    parser.add_argument(
        "--probes",
        "-p",
        type=str,
        default="all",
        help="comma-separated list of probe names to use, or 'all' for all available probes (default: all). Examples: 'dan', 'encoding', 'owasp.LLMTopTen'",
    )
    parser.add_argument(
        "--probe_tags",
        type=str,
        help="only include probes with a tag that starts with this value (e.g., 'owasp:llm01', 'avid-effect')",
    )
    parser.add_argument(
        "--probe_module",
        type=str,
        help="load all probes from a specific module (e.g., 'dan', 'encoding', 'lmrc')",
    )
    parser.add_argument(
        "--benign_prompts",
        action="store_true",
        help="generate category-specific benign prompts using LLM (same count as adversarial per category). Requires benign_llm config in detectors.yaml.",
    )
    parser.add_argument(
        "--include_inactive",
        action="store_true",
        help="include inactive 'Full' probes (e.g., DanInTheWildFull with 665 prompts instead of DanInTheWild with 64). WARNING: This can generate 10,000+ prompts and take much longer to run.",
    )
    parser.add_argument(
        "--no_deduplicate",
        action="store_true",
        help="disable deduplication of prompts within each probe. By default, duplicate prompts are removed to avoid redundant testing. Use this flag to keep all prompts including duplicates.",
    )

    # REPORTING
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=None,
        help="organize results by taxonomy (e.g., 'owasp', 'avid-effect'). Results will mirror the probe structure.",
    )

    # COMMANDS
    parser.add_argument(
        "--list_probes",
        action="store_true",
        help="list all available garak probes that can be used for testing",
    )
    parser.add_argument(
        "--list_detectors",
        action="store_true",
        help="list configured detectors from the YAML file",
    )

    args = parser.parse_args(arguments)

    # Start logging
    log_filename = start_logging(args.output, args.verbose)

    # Handle commands
    try:
        if args.list_probes:
            print_probes(probe_filter=args.probes, tag_filter=args.probe_tags)

        elif args.list_detectors:
            if not args.detectors:
                print("❌ Error: --detectors config file required for --list_detectors")
                sys.exit(1)
            print_detectors_config(args.detectors)

        elif args.detectors:
            # Run benchmark
            print(f"📜 logging to {log_filename}")
            run_benchmark(args)

        else:
            print("nothing to do 🤷  try --help")
            print("💡 typical usage: guardrails_eval.py --detectors config.yaml --probes all")

    except KeyboardInterrupt:
        print("\n⚠️  User cancel received, terminating")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
