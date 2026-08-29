"""Zipf's Law estimation, Mandelbrot generalized fitting, and residual analysis."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import optimize, stats

from src.frequency_analysis import compute_frequency_distribution


def fit_classic_zipf(
    ranked_data: List[dict],
    min_rank: int = 2,
    max_rank: Optional[int] = None,
) -> dict:
    """Fit standard linear log-log Zipf model: log(f) = log(C) - s * log(r)."""
    selected = [d for d in ranked_data if d["rank"] >= min_rank and (max_rank is None or d["rank"] <= max_rank)]
    if len(selected) < 3:
        return {
            "model": "classic_zipf",
            "zipf_exponent": None,
            "intercept": None,
            "r2": None,
            "rmse": None,
            "slope_std_err": None,
            "sample_size": len(selected),
            "fit_min_rank": min_rank,
            "fit_max_rank": max_rank,
        }

    log_ranks = np.array([d["log_rank"] for d in selected], dtype=float)
    log_freqs = np.array([d["log_frequency"] for d in selected], dtype=float)

    reg = stats.linregress(log_ranks, log_freqs)
    predicted = reg.slope * log_ranks + reg.intercept
    residuals = log_freqs - predicted
    
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_freqs - log_freqs.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    rmse = math.sqrt(ss_res / len(log_freqs))

    return {
        "model": "classic_zipf",
        "zipf_exponent": round(float(-reg.slope), 4),
        "intercept": round(float(reg.intercept), 4),
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "slope_std_err": round(float(reg.stderr), 5) if reg.stderr else 0.0,
        "sample_size": len(selected),
        "fit_min_rank": min_rank,
        "fit_max_rank": max_rank,
    }


def mandelbrot_func(rank, log_C, s, beta):
    return log_C - s * np.log(rank + beta)


def fit_zipf_mandelbrot(
    ranked_data: List[dict],
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> dict:
    """Fit Zipf-Mandelbrot model: log(f) = log(C) - s * log(r + beta)."""
    selected = [d for d in ranked_data if d["rank"] >= min_rank and (max_rank is None or d["rank"] <= max_rank)]
    if len(selected) < 4:
        return {
            "model": "zipf_mandelbrot",
            "mandelbrot_s": None,
            "mandelbrot_beta": None,
            "mandelbrot_r2": None,
        }

    ranks = np.array([d["rank"] for d in selected], dtype=float)
    log_freqs = np.array([d["log_frequency"] for d in selected], dtype=float)

    try:
        # Initial guess: log_C = max(log_freqs), s = 1.0, beta = 2.0
        popt, _ = optimize.curve_fit(
            mandelbrot_func,
            ranks,
            log_freqs,
            p0=[max(log_freqs), 1.0, 1.0],
            bounds=([-np.inf, 0.01, 0.0], [np.inf, 5.0, 500.0]),
            maxfev=5000,
        )
        log_C_est, s_est, beta_est = popt
        pred = mandelbrot_func(ranks, log_C_est, s_est, beta_est)
        ss_res = float(np.sum((log_freqs - pred) ** 2))
        ss_tot = float(np.sum((log_freqs - log_freqs.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return {
            "model": "zipf_mandelbrot",
            "mandelbrot_s": round(float(s_est), 4),
            "mandelbrot_beta": round(float(beta_est), 4),
            "mandelbrot_log_C": round(float(log_C_est), 4),
            "mandelbrot_r2": round(float(r2), 4),
        }
    except Exception:
        return {
            "model": "zipf_mandelbrot",
            "mandelbrot_s": None,
            "mandelbrot_beta": None,
            "mandelbrot_r2": None,
        }


def analyze_zipf_behavior(
    tokens: List[str],
    min_rank: int = 2,
    max_rank: Optional[int] = None,
) -> Tuple[dict, List[dict]]:
    """Complete Zipfian analysis combining classic OLS and Mandelbrot non-linear fitting."""
    ranked_data, counts = compute_frequency_distribution(tokens)
    classic = fit_classic_zipf(ranked_data, min_rank=min_rank, max_rank=max_rank)
    mandelbrot = fit_zipf_mandelbrot(ranked_data, min_rank=1, max_rank=max_rank)
    
    # Head vs Body vs Tail R2
    head_fit = fit_classic_zipf(ranked_data, min_rank=1, max_rank=min(50, len(ranked_data)))
    tail_fit = fit_classic_zipf(ranked_data, min_rank=min(50, len(ranked_data)), max_rank=None)

    combined_stats = {
        **classic,
        **mandelbrot,
        "head_r2": head_fit.get("r2"),
        "head_zipf_exponent": head_fit.get("zipf_exponent"),
        "tail_r2": tail_fit.get("r2"),
        "tail_zipf_exponent": tail_fit.get("zipf_exponent"),
    }
    return combined_stats, ranked_data
