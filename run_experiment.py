"""CLI Entry Point for Zipf's Law and Tokenization Research Framework."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from experiments.run_all import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Zipf's Law and Multilingual Tokenization Research Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="demo",
        help="Execution mode: 'demo' for fast sample run, 'full' for complete large-scale experiments.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    results = run_pipeline(mode=args.mode, config_path=args.config)
    print(f"\nExecution Summary:")
    print(f"  * Languages evaluated: {len(results['corpus_metadata'])}")
    print(f"  * Pretrained tokenizer configurations: {len(results['tokenizer_benchmarks'])}")
    print(f"  * Controlled vocabulary sweep points: {len(results['vocabulary_sweep'])}")
    print(f"  * Sweet-spot candidate profiles: {len(results['sweet_spots'])}")


if __name__ == "__main__":
    main()
