"""Experiment 1: Multilingual Word and Subword Zipf-Law Behavior Comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.preprocessing import extract_words
from src.frequency_analysis import compute_distribution_metrics
from src.zipf_analysis import analyze_zipf_behavior
from src.utils import plot_zipf_single, plot_zipf_multilingual, save_csv, save_json


def run_language_comparison_experiment(
    corpora: Dict[str, Dict[str, str]],
    config: dict,
    results_dir: Path = Path("results"),
) -> Tuple[List[dict], Dict[str, Tuple[List[dict], dict]]]:
    """Execute multilingual word-level and baseline token Zipfian comparison."""
    min_rank = config.get("fit_min_rank", 2)
    max_rank = config.get("fit_max_rank", None)

    summary_rows = []
    lang_ranked_data = {}
    plots_dir = results_dir / "plots"
    raw_dir = results_dir / "raw"

    for lang, corp in corpora.items():
        text = corp["full"]
        words_list = extract_words(text)
        
        # Analyze Zipf behavior on word-level tokens
        zipf_stats, ranked_data = analyze_zipf_behavior(words_list, min_rank=min_rank, max_rank=max_rank)
        dist_metrics = compute_distribution_metrics(words_list)
        
        lang_ranked_data[lang] = (ranked_data, zipf_stats)

        # Plot individual language Zipf curve
        plot_zipf_single(
            ranked_data=ranked_data,
            stats=zipf_stats,
            language=lang,
            output_path=plots_dir / f"zipf_{lang}.png",
            title_suffix="(Word-Level)",
        )

        # Save raw JSON
        save_json(raw_dir / f"{lang}_word_zipf.json", {
            "language": lang,
            "distribution_metrics": dist_metrics,
            "zipf_statistics": zipf_stats,
            "top_20_tokens": ranked_data[:20],
        })

        row = {
            "language": lang,
            "source": corp.get("source", "Wikipedia"),
            "total_words": len(words_list),
            "unique_words": dist_metrics["unique_tokens"],
            "shannon_entropy": dist_metrics["shannon_entropy_bits"],
            "hapax_ratio": dist_metrics["hapax_ratio"],
            "zipf_exponent_s": zipf_stats.get("zipf_exponent"),
            "zipf_r2": zipf_stats.get("r2"),
            "zipf_rmse": zipf_stats.get("rmse"),
            "mandelbrot_beta": zipf_stats.get("mandelbrot_beta"),
            "mandelbrot_s": zipf_stats.get("mandelbrot_s"),
            "mandelbrot_r2": zipf_stats.get("mandelbrot_r2"),
            "head_r2": zipf_stats.get("head_r2"),
            "tail_r2": zipf_stats.get("tail_r2"),
        }
        summary_rows.append(row)

    # Combined multilingual plot
    plot_zipf_multilingual(lang_ranked_data, plots_dir / "zipf_multilingual_comparison.png")
    
    # Save CSV summary
    save_csv(results_dir / "tables" / "language_comparison.csv", summary_rows)
    return summary_rows, lang_ranked_data
