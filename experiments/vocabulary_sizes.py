"""Experiment 3: Controlled Vocabulary-Size Sweep, Distributional Stability, and Sweet-Spot Identification."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from src.preprocessing import extract_words
from src.tokenizer_training import train_controlled_bpe
from src.frequency_analysis import compute_distribution_metrics
from src.zipf_analysis import analyze_zipf_behavior
from src.stability_analysis import compute_step_stability
from src.sweet_spot import analyze_vocabulary_sweet_spot
from src.utils import plot_vocab_sweep_curves, save_csv, save_json


def run_vocabulary_sizes_experiment(
    corpora: Dict[str, Dict[str, str]],
    mode: str,
    config: dict,
    results_dir: Path = Path("results"),
) -> Tuple[List[dict], List[dict]]:
    """Execute strictly controlled BPE vocabulary size sweep across English, Hindi, and Arabic."""
    vocab_sizes = config["vocab_sizes_demo"] if mode == "demo" else config["vocab_sizes_full"]
    thresholds = config.get("thresholds", {})
    min_rank = config.get("fit_min_rank", 2)
    max_rank = config.get("fit_max_rank", None)

    vocab_experiment_rows: List[dict] = []
    sweet_spot_rows: List[dict] = []
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for lang, corp in corpora.items():
        train_text = corp["train"]
        val_text = corp["val"]
        full_text = corp["full"]

        train_words = extract_words(train_text)
        val_words = extract_words(val_text)
        full_words = extract_words(full_text)

        num_train_words = max(1, len(train_words))
        num_val_words = max(1, len(val_words))
        num_full_words = max(1, len(full_words))
        num_full_chars = max(1, len(full_text))

        prev_row = None
        prev_counts = None

        for target_v in vocab_sizes:
            # 1. Train controlled BPE model on training split
            tokenizer = train_controlled_bpe(
                text=train_text,
                target_vocab_size=target_v,
                language=lang,
                save_dir=models_dir,
            )

            # 2. Tokenize full corpus
            full_tokens = tokenizer.encode(full_text)
            val_tokens = tokenizer.encode(val_text)

            curr_counts = Counter(full_tokens)
            dist_metrics = compute_distribution_metrics(full_tokens, target_vocab_size=target_v)
            zipf_stats, ranked_data = analyze_zipf_behavior(full_tokens, min_rank=min_rank, max_rank=max_rank)

            tpw = len(full_tokens) / num_full_words
            tpc = len(full_tokens) / num_full_chars
            val_tpw = len(val_tokens) / num_val_words

            avg_token_len = sum(len(t.replace("</w>", "").replace("Ġ", "")) for t in full_tokens) / max(1, len(full_tokens))

            # Current base metrics
            curr_row = {
                "language": lang,
                "tokenizer_type": "controlled_bpe",
                "requested_vocab_size": target_v,
                "actual_vocab_size": tokenizer.get_actual_vocab_size(),
                "corpus_tokens": len(full_tokens),
                "unique_tokens_used": dist_metrics["unique_tokens"],
                "vocabulary_utilization": dist_metrics["vocabulary_utilization"],
                "tokens_per_word": round(tpw, 4),
                "tokens_per_character": round(tpc, 4),
                "val_tokens_per_word": round(val_tpw, 4),
                "average_token_length": round(avg_token_len, 2),
                "shannon_entropy": dist_metrics["shannon_entropy_bits"],
                "zipf_exponent": zipf_stats.get("zipf_exponent"),
                "zipf_intercept": zipf_stats.get("intercept"),
                "r2": zipf_stats.get("r2"),
                "rmse": zipf_stats.get("rmse"),
                "mandelbrot_beta": zipf_stats.get("mandelbrot_beta"),
                "mandelbrot_s": zipf_stats.get("mandelbrot_s"),
                "mandelbrot_r2": zipf_stats.get("mandelbrot_r2"),
                "head_r2": zipf_stats.get("head_r2"),
                "tail_r2": zipf_stats.get("tail_r2"),
            }

            # Compute transitional stability deltas
            stability = compute_step_stability(prev_row, curr_row, prev_counts, curr_counts)
            curr_row.update(stability)

            vocab_experiment_rows.append(curr_row)
            prev_row = curr_row
            prev_counts = curr_counts

        # Perform sweet-spot analysis for this language
        sweet_spot = analyze_vocabulary_sweet_spot(lang, vocab_experiment_rows, thresholds)
        sweet_spot_rows.append(sweet_spot)

    # Save Tables
    save_csv(results_dir / "tables" / "vocabulary_experiment.csv", vocab_experiment_rows)
    save_csv(results_dir / "tables" / "sweet_spots.csv", sweet_spot_rows)

    # Save Raw JSON
    save_json(results_dir / "raw" / "vocabulary_experiment_raw.json", {
        "mode": mode,
        "vocabulary_sizes": vocab_sizes,
        "vocab_experiment_rows": vocab_experiment_rows,
        "sweet_spots": sweet_spot_rows,
    })

    # Generate Full Plot Suite
    plot_vocab_sweep_curves(vocab_experiment_rows, sweet_spot_rows, results_dir / "plots")

    return vocab_experiment_rows, sweet_spot_rows
