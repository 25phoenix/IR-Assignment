"""Master orchestration pipeline executing all three experimental phases and generating the final report."""
from __future__ import annotations

import sys
# Set UTF-8 encoding on standard streams to avoid Windows charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import Dict, Any

from src.config import load_config
from src.data_collection import prepare_datasets
from experiments.language_comparison import run_language_comparison_experiment
from experiments.existing_tokenizers import run_existing_tokenizers_experiment
from experiments.vocabulary_sizes import run_vocabulary_sizes_experiment
from report.generate_report import generate_research_report
from src.utils import save_json, save_csv


def run_pipeline(mode: str = "demo", config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute the complete end-to-end research pipeline."""
    print("=" * 70)
    print(f"🚀 STARTING ZIPF'S LAW & TOKENIZATION RESEARCH PIPELINE (MODE: {mode.upper()})")
    print("=" * 70)

    config = load_config(config_path)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Preparation
    print("\n[Phase 1/5] Acquiring and Preparing Multilingual Datasets (English, Hindi, Arabic)...")
    corpora, corpus_metadata = prepare_datasets(
        languages=config["languages"],
        mode=mode,
        max_chars_demo=config["max_chars_demo"],
        max_chars_full=config["max_chars_full"],
        train_val_split_ratio=config["train_val_split"],
        download_wikipedia=config.get("download_wikipedia", True),
    )
    save_csv(results_dir / "tables" / "corpus_metadata.csv", corpus_metadata)
    for meta in corpus_metadata:
        print(f"  - [{meta['language'].upper()}]: {meta['words']} words, {meta['characters']} chars | Source: {meta['source']}")

    # 2. Experiment 1: Word-Level & Baseline Zipf Comparison
    print("\n[Phase 2/5] Running Experiment 1: Word-Level Multilingual Zipf's Law Analysis...")
    lang_summary, lang_ranked = run_language_comparison_experiment(corpora, config, results_dir)
    for r in lang_summary:
        print(f"  - {r['language'].capitalize()}: Zipf s={r['zipf_exponent_s']}, R²={r['zipf_r2']} | Mandelbrot β={r['mandelbrot_beta']}, s={r['mandelbrot_s']}")

    # 3. Experiment 2: Pretrained Tokenizers Benchmark
    print("\n[Phase 3/5] Running Experiment 2: Pretrained Tokenizer Benchmarking (LLaMA, Qwen, DeepSeek, GPT-2, Tiktoken)...")
    tok_rows, seg_rows = run_existing_tokenizers_experiment(corpora, config, results_dir)
    print(f"  - Evaluated {len(tok_rows)} language × tokenizer combinations successfully.")
    for r in tok_rows[:6]:
        if r.get("status") == "success":
            print(f"    * {r['language'].capitalize()} × {r['tokenizer']} (Vocab {r['vocabulary_size']}): TPW={r['tokens_per_word']}, s={r['zipf_exponent']}, R²={r['zipf_r2']}")

    # 4. Experiment 3: Controlled Vocabulary-Size Sweep & Sweet Spot
    print(f"\n[Phase 4/5] Running Experiment 3: Controlled BPE Vocabulary Sweep & Sweet-Spot Identification...")
    vocab_rows, sweet_spots = run_vocabulary_sizes_experiment(corpora, mode, config, results_dir)
    print("  - Sweet-Spot Consensus Results:")
    for s in sweet_spots:
        print(f"    * {s['language'].capitalize()}: Candidate Vocab = {s['candidate_vocab_size']} (Confidence: {s['confidence']}) | Rationale: {s['rationale']}")

    # 5. Report Generation
    print("\n[Phase 5/5] Generating Comprehensive Scientific Research Report...")
    generate_research_report(mode=mode, results_dir=results_dir, report_path=Path("report/report.md"))
    print("  - Saved report to `report/report.md`")

    # Save Run Metadata
    save_json(results_dir / "raw" / "run_metadata.json", {
        "mode": mode,
        "config": config,
        "corpus_metadata": corpus_metadata,
        "sweet_spots": sweet_spots,
    })

    print("\n" + "=" * 70)
    print("✅ EXPERIMENTAL PIPELINE EXECUTED SUCCESSFULLY WITH FULL REPRODUCIBILITY!")
    print("=" * 70)

    return {
        "corpus_metadata": corpus_metadata,
        "language_summary": lang_summary,
        "tokenizer_benchmarks": tok_rows,
        "vocabulary_sweep": vocab_rows,
        "sweet_spots": sweet_spots,
    }
