"""Distributional stability metrics: Jensen-Shannon Divergence, TV Distance, and Rank Correlation."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def compute_js_divergence(p_counts: Counter, q_counts: Counter) -> float:
    """Calculate the symmetric Jensen-Shannon divergence in bits [0, 1]."""
    all_keys = set(p_counts.keys()) | set(q_counts.keys())
    total_p = sum(p_counts.values())
    total_q = sum(q_counts.values())
    
    if total_p == 0 or total_q == 0 or not all_keys:
        return 0.0

    p = np.array([p_counts.get(k, 0) / total_p for k in all_keys], dtype=float)
    q = np.array([q_counts.get(k, 0) / total_q for k in all_keys], dtype=float)
    m = 0.5 * (p + q)

    # Safe KL divergence calculation
    p_mask = (p > 0) & (m > 0)
    q_mask = (q > 0) & (m > 0)
    kl_pm = float(np.sum(p[p_mask] * np.log2(p[p_mask] / m[p_mask])))
    kl_qm = float(np.sum(q[q_mask] * np.log2(q[q_mask] / m[q_mask])))

    js = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(js, 0.0, 1.0))


def compute_total_variation_distance(p_counts: Counter, q_counts: Counter) -> float:
    """Calculate Total Variation Distance between two probability distributions."""
    all_keys = set(p_counts.keys()) | set(q_counts.keys())
    total_p = sum(p_counts.values())
    total_q = sum(q_counts.values())
    
    if total_p == 0 or total_q == 0 or not all_keys:
        return 0.0

    tv = 0.5 * sum(
        abs((p_counts.get(k, 0) / total_p) - (q_counts.get(k, 0) / total_q))
        for k in all_keys
    )
    return float(np.clip(tv, 0.0, 1.0))


def compute_rank_correlation(p_counts: Counter, q_counts: Counter, top_k: int = 100) -> Tuple[float, float]:
    """Compute Spearman and Kendall rank correlations on the top-k most frequent shared tokens."""
    shared_tokens = list(set(p_counts.keys()) & set(q_counts.keys()))
    if len(shared_tokens) < 5:
        return 0.0, 0.0

    # Sort shared tokens by frequency in P
    shared_tokens.sort(key=lambda t: p_counts[t], reverse=True)
    eval_tokens = shared_tokens[:top_k]

    ranks_p = [p_counts[t] for t in eval_tokens]
    ranks_q = [q_counts[t] for t in eval_tokens]

    spearman_corr, _ = stats.spearmanr(ranks_p, ranks_q)
    kendall_corr, _ = stats.kendalltau(ranks_p, ranks_q)

    return (
        float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        float(kendall_corr) if not np.isnan(kendall_corr) else 0.0,
    )


def compute_step_stability(
    prev_row: Optional[dict],
    curr_row: dict,
    prev_counts: Optional[Counter],
    curr_counts: Counter,
) -> dict:
    """Compute all transitional delta and divergence metrics between consecutive vocabulary sizes."""
    if prev_row is None or prev_counts is None:
        return {
            "js_divergence": None,
            "tv_distance": None,
            "spearman_rank_corr": None,
            "delta_zipf_exponent": None,
            "delta_r2": None,
            "relative_efficiency_gain": None,
            "marginal_elasticity": None,
        }

    js = compute_js_divergence(prev_counts, curr_counts)
    tv = compute_total_variation_distance(prev_counts, curr_counts)
    spearman, _ = compute_rank_correlation(prev_counts, curr_counts)

    # Zipf delta
    prev_s = prev_row.get("zipf_exponent") or 0.0
    curr_s = curr_row.get("zipf_exponent") or 0.0
    delta_s = abs(curr_s - prev_s)

    # R2 delta
    prev_r2 = prev_row.get("r2") or 0.0
    curr_r2 = curr_row.get("r2") or 0.0
    delta_r2 = abs(curr_r2 - prev_r2)

    # Efficiency gain (reduction in tokens per word)
    prev_tpw = prev_row.get("tokens_per_word") or 1.0
    curr_tpw = curr_row.get("tokens_per_word") or 1.0
    rel_eff_gain = (prev_tpw - curr_tpw) / max(prev_tpw, 1e-9)

    # Marginal Elasticity: % change in TPW divided by % change in Vocab
    prev_v = prev_row.get("requested_vocab_size") or 1
    curr_v = curr_row.get("requested_vocab_size") or 1
    pct_v_change = (curr_v - prev_v) / max(prev_v, 1)
    marginal_elasticity = (rel_eff_gain / max(pct_v_change, 1e-9)) if pct_v_change > 0 else 0.0

    return {
        "js_divergence": round(float(js), 5),
        "tv_distance": round(float(tv), 5),
        "spearman_rank_corr": round(float(spearman), 4),
        "delta_zipf_exponent": round(float(delta_s), 4),
        "delta_r2": round(float(delta_r2), 4),
        "relative_efficiency_gain": round(float(rel_eff_gain), 4),
        "marginal_elasticity": round(float(marginal_elasticity), 4),
    }
