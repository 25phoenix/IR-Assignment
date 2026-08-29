"""Experiment 2: Pretrained Tokenizer Evaluation across English, Hindi, and Arabic."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.tokenizer_comparison import PretrainedTokenizerBenchmark
from src.utils import plot_tokenizer_tax_bars, save_csv, save_json


def run_existing_tokenizers_experiment(
    corpora: Dict[str, Dict[str, str]],
    config: dict,
    results_dir: Path = Path("results"),
) -> Tuple[List[dict], List[dict]]:
    """Execute pretrained tokenizer benchmark on LLaMA, Qwen, DeepSeek, GPT-2, and Tiktoken."""
    tok_dict = config.get("pretrained_tokenizers", {})
    min_rank = config.get("fit_min_rank", 2)
    max_rank = config.get("fit_max_rank", None)

    benchmark = PretrainedTokenizerBenchmark(tok_dict)
    benchmark_rows, segmentation_samples = benchmark.benchmark_corpora(
        corpora=corpora,
        min_rank=min_rank,
        max_rank=max_rank,
    )

    # Save CSV outputs
    save_csv(results_dir / "tables" / "existing_tokenizers.csv", benchmark_rows)
    save_csv(results_dir / "tables" / "parallel_segmentation_samples.csv", segmentation_samples)

    # Plot bar chart for Tokenizer Tax
    plot_tokenizer_tax_bars(benchmark_rows, results_dir / "plots" / "tokenizer_tax_fertility.png")

    # Save raw metadata
    save_json(results_dir / "raw" / "pretrained_tokenizers_benchmark.json", {
        "loaded_models": list(benchmark.loaded_tokenizers.keys()),
        "results": benchmark_rows,
    })

    return benchmark_rows, segmentation_samples
