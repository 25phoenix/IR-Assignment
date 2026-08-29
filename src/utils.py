"""Plotting utilities, CSV/JSON serialization, and reporting helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling parameters
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
})

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def save_csv(output_path: str | Path, rows: List[dict]) -> None:
    """Save list of dictionaries to CSV with header."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(output_path: str | Path, data: Any) -> None:
    """Save data to JSON with UTF-8 encoding and indentation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_zipf_single(
    ranked_data: List[dict],
    stats: dict,
    language: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot rank vs frequency with linear OLS fit line on log-log axes."""
    plt.figure(figsize=(7, 5))
    ranks = [d["rank"] for d in ranked_data]
    freqs = [d["frequency"] for d in ranked_data]
    
    plt.loglog(ranks, freqs, "o", markersize=3, alpha=0.6, label="Observed Tokens", color="#1f77b4")
    
    # Plot fitted line
    s = stats.get("zipf_exponent")
    intercept = stats.get("intercept")
    r2 = stats.get("r2")
    
    if s is not None and intercept is not None:
        fit_ranks = np.array(ranks[1:])  # from rank 2
        pred_freqs = np.exp(intercept) * (fit_ranks ** (-s))
        label_text = f"OLS Fit: $f(r) \\propto r^{{-{s:.2f}}}$ ($R^2={r2:.3f}$)"
        plt.loglog(fit_ranks, pred_freqs, "r--", linewidth=2, label=label_text)
        
    plt.xlabel("Rank $r$ (log scale)")
    plt.ylabel("Frequency $f(r)$ (log scale)")
    plt.title(f"Zipf's Law Distribution: {language.capitalize()} {title_suffix}".strip())
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_zipf_multilingual(
    language_ranked_data: Dict[str, Tuple[List[dict], dict]],
    output_path: Path,
) -> None:
    """Plot cross-lingual comparison of rank-frequency curves on a single log-log plot."""
    plt.figure(figsize=(8, 5.5))
    
    for idx, (lang, (ranked, stats)) in enumerate(language_ranked_data.items()):
        ranks = [d["rank"] for d in ranked]
        freqs = [d["frequency"] for d in ranked]
        s = stats.get("zipf_exponent", 0.0)
        r2 = stats.get("r2", 0.0)
        c = COLORS[idx % len(COLORS)]
        plt.loglog(ranks, freqs, "-", linewidth=2, alpha=0.85, color=c, label=f"{lang.capitalize()} ($s={s:.2f}, R^2={r2:.2f}$)")

    plt.xlabel("Rank $r$ (log scale)")
    plt.ylabel("Frequency $f(r)$ (log scale)")
    plt.title("Cross-Linguistic Zipf Distribution (English vs Hindi vs Arabic)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_tokenizer_tax_bars(
    benchmark_rows: List[dict],
    output_path: Path,
) -> None:
    """Plot bar chart showing Tokens Per Word across models and languages (The Tokenizer Tax)."""
    valid_rows = [r for r in benchmark_rows if r.get("status") == "success"]
    if not valid_rows:
        return

    languages = sorted(list({r["language"] for r in valid_rows}))
    tokenizers = sorted(list({r["tokenizer"] for r in valid_rows}))
    
    x = np.arange(len(languages))
    width = 0.8 / max(1, len(tokenizers))
    
    plt.figure(figsize=(9, 5.5))
    for i, tok in enumerate(tokenizers):
        tok_tpw = []
        for lang in languages:
            match = next((r for r in valid_rows if r["language"] == lang and r["tokenizer"] == tok), None)
            tok_tpw.append(match["tokens_per_word"] if match else 0.0)
        offset = (i - len(tokenizers) / 2 + 0.5) * width
        plt.bar(x + offset, tok_tpw, width, label=tok, color=COLORS[i % len(COLORS)], alpha=0.85)

    plt.xlabel("Language")
    plt.ylabel("Tokens Per Word (Fertility)")
    plt.title("Cross-Linguistic Tokenizer Fertility (The 'Tokenizer Tax')")
    plt.xticks(x, [l.capitalize() for l in languages])
    plt.grid(axis="y", ls=":", alpha=0.5)
    plt.legend(title="Tokenizer Model")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_vocab_sweep_curves(
    vocab_rows: List[dict],
    sweet_spots: List[dict],
    plots_dir: Path,
) -> None:
    """Generate the full suite of vocabulary size sweep plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    languages = sorted(list({r["language"] for r in vocab_rows}))
    
    # 1. Vocab Size vs Tokens Per Word
    plt.figure(figsize=(8, 5))
    for idx, lang in enumerate(languages):
        rows = [r for r in vocab_rows if r["language"] == lang]
        x = [r["requested_vocab_size"] for r in rows]
        y = [r["tokens_per_word"] for r in rows]
        c = COLORS[idx % len(COLORS)]
        plt.plot(x, y, "o-", linewidth=2, color=c, label=lang.capitalize())
        
        # Mark sweet spot
        cand = next((s["candidate_vocab_size"] for s in sweet_spots if s["language"] == lang), None)
        if cand and cand in x:
            cand_y = next(r["tokens_per_word"] for r in rows if r["requested_vocab_size"] == cand)
            plt.scatter([cand], [cand_y], s=120, color=c, edgecolors="black", zorder=5, label=f"{lang.capitalize()} Sweet Spot ({cand})")
            
    plt.xscale("log")
    plt.xlabel("Vocabulary Size Budget $V$ (log scale)")
    plt.ylabel("Tokens Per Word (Fertility)")
    plt.title("Vocabulary Size vs Tokenization Fertility (Compression Curve)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vocab_vs_tokens_per_word.png", dpi=200)
    plt.close()

    # 2. Vocab Size vs Zipf Exponent
    plt.figure(figsize=(8, 5))
    for idx, lang in enumerate(languages):
        rows = [r for r in vocab_rows if r["language"] == lang and r.get("zipf_exponent") is not None]
        x = [r["requested_vocab_size"] for r in rows]
        y = [r["zipf_exponent"] for r in rows]
        plt.plot(x, y, "s-", linewidth=2, color=COLORS[idx % len(COLORS)], label=lang.capitalize())
    plt.xscale("log")
    plt.xlabel("Vocabulary Size Budget $V$ (log scale)")
    plt.ylabel("Fitted Zipf Exponent $s$")
    plt.title("Vocabulary Size vs Zipf Slope Parameter $s$")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vocab_vs_zipf_exponent.png", dpi=200)
    plt.close()

    # 3. Vocab Size vs R2 Goodness-of-Fit
    plt.figure(figsize=(8, 5))
    for idx, lang in enumerate(languages):
        rows = [r for r in vocab_rows if r["language"] == lang and r.get("r2") is not None]
        x = [r["requested_vocab_size"] for r in rows]
        y = [r["r2"] for r in rows]
        plt.plot(x, y, "^-", linewidth=2, color=COLORS[idx % len(COLORS)], label=lang.capitalize())
    plt.xscale("log")
    plt.xlabel("Vocabulary Size Budget $V$ (log scale)")
    plt.ylabel("Zipf OLS Coefficient of Determination $R^2$")
    plt.title("Vocabulary Size vs Zipf Linearity Fit $R^2$")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vocab_vs_r2.png", dpi=200)
    plt.close()

    # 4. Vocab Size vs Jensen-Shannon Divergence
    plt.figure(figsize=(8, 5))
    for idx, lang in enumerate(languages):
        rows = [r for r in vocab_rows if r["language"] == lang and r.get("js_divergence") is not None]
        x = [r["requested_vocab_size"] for r in rows]
        y = [r["js_divergence"] for r in rows]
        plt.plot(x, y, "d--", linewidth=2, color=COLORS[idx % len(COLORS)], label=lang.capitalize())
    plt.xscale("log")
    plt.xlabel("Vocabulary Size Budget $V$ (log scale)")
    plt.ylabel("Jensen-Shannon Divergence $D_{JS}(P_{V_{i-1}} || P_{V_i})$")
    plt.title("Vocabulary Growth vs Distributional Stationarity")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vocab_vs_js_divergence.png", dpi=200)
    plt.close()

    # 5. Marginal Compression Gain vs Vocab Size
    plt.figure(figsize=(8, 5))
    for idx, lang in enumerate(languages):
        rows = [r for r in vocab_rows if r["language"] == lang and r.get("relative_efficiency_gain") is not None]
        x = [r["requested_vocab_size"] for r in rows]
        y = [r["relative_efficiency_gain"] for r in rows]
        plt.plot(x, y, "x-", linewidth=2, color=COLORS[idx % len(COLORS)], label=lang.capitalize())
    plt.xscale("log")
    plt.xlabel("Vocabulary Size Budget $V$ (log scale)")
    plt.ylabel("Marginal Compression Gain $\\Delta \\text{TPW} / \\text{TPW}$")
    plt.title("Diminishing Marginal Returns of Vocabulary Expansion")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "vocab_vs_marginal_gain.png", dpi=200)
    plt.close()
