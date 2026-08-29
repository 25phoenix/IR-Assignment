"""Analytical criteria and algorithms for vocabulary size sweet-spot detection."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


def detect_kneedle_elbow(x_vals: List[float], y_vals: List[float]) -> Optional[int]:
    """
    Kneedle algorithm for automatic elbow/knee detection on diminishing return curves.
    Finds the index of the point with maximum perpendicular distance to the line connecting endpoints.
    """
    if len(x_vals) < 3 or len(y_vals) < 3:
        return None

    # Normalize x (log scale) and y to [0, 1]
    x_arr = np.log2(np.array(x_vals, dtype=float))
    y_arr = np.array(y_vals, dtype=float)

    x_norm = (x_arr - x_arr.min()) / max(1e-9, x_arr.max() - x_arr.min())
    y_norm = (y_arr - y_arr.min()) / max(1e-9, y_arr.max() - y_arr.min())

    # Endpoints
    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    denom = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if denom < 1e-9:
        return None

    distances = [
        abs((y2 - y1) * x_norm[i] - (x2 - x1) * y_norm[i] + x2 * y1 - y2 * x1) / denom
        for i in range(len(x_norm))
    ]

    max_idx = int(np.argmax(distances))
    return max_idx


def compute_pareto_cost(
    vocab_size: int,
    tokens_per_word: float,
    baseline_tpw: float,
    d_model: int = 4096,
    lambda_param: float = 0.00005,
) -> float:
    """Compute bi-criterion objective balancing sequence length against embedding parameter overhead."""
    fertility_cost = tokens_per_word / max(1e-9, baseline_tpw)
    param_cost = lambda_param * (vocab_size * d_model) / 100000.0
    return float(fertility_cost + param_cost)


def analyze_vocabulary_sweet_spot(
    language: str,
    vocab_rows: List[dict],
    thresholds: dict,
) -> dict:
    """
    Comprehensive multi-criterion sweet spot determination for a given language.
    Combines:
    1. Multi-factor threshold criteria (diminishing fertility + JS divergence + Zipf stability)
    2. Kneedle geometric elbow detection on the TPW curve
    3. Bi-criterion Pareto optimization cost
    """
    lang_rows = [r for r in vocab_rows if r["language"] == language]
    if not lang_rows:
        return {"language": language, "candidate_vocab_size": None, "status": "no data"}

    # 1. Multi-factor threshold candidate
    th_candidate = None
    th_reasons = []
    for i in range(1, len(lang_rows)):
        curr = lang_rows[i]
        prior = lang_rows[i - 1]

        eff_gain = curr.get("relative_efficiency_gain")
        js = curr.get("js_divergence")
        d_s = curr.get("delta_zipf_exponent")
        d_r2 = curr.get("delta_r2")
        elast = curr.get("marginal_elasticity")

        checks = [
            eff_gain is not None and abs(eff_gain) <= thresholds.get("relative_efficiency_gain", 0.03),
            js is not None and js <= thresholds.get("js_divergence", 0.05),
            d_s is not None and d_s <= thresholds.get("zipf_delta", 0.05),
            d_r2 is not None and d_r2 <= thresholds.get("r2_delta", 0.02),
        ]

        if all(checks):
            th_candidate = curr["requested_vocab_size"]
            th_reasons = [
                f"Relative compression gain ({eff_gain:.3f}) <= {thresholds.get('relative_efficiency_gain')}",
                f"JS divergence ({js:.4f}) <= {thresholds.get('js_divergence')}",
                f"Zipf slope delta ({d_s:.4f}) <= {thresholds.get('zipf_delta')}",
                f"R2 delta ({d_r2:.4f}) <= {thresholds.get('r2_delta')}",
            ]
            break

    # 2. Kneedle geometric elbow detection
    vocab_sizes = [r["requested_vocab_size"] for r in lang_rows]
    tpws = [r["tokens_per_word"] for r in lang_rows]
    knee_idx = detect_kneedle_elbow(vocab_sizes, tpws)
    kneedle_candidate = vocab_sizes[knee_idx] if knee_idx is not None else None

    # 3. Pareto optimization candidate
    baseline_tpw = lang_rows[0]["tokens_per_word"]
    pareto_costs = [
        compute_pareto_cost(r["requested_vocab_size"], r["tokens_per_word"], baseline_tpw)
        for r in lang_rows
    ]
    pareto_best_idx = int(np.argmin(pareto_costs))
    pareto_candidate = vocab_sizes[pareto_best_idx]

    # Consensus candidate: prefer threshold if found, else Kneedle elbow
    primary_candidate = th_candidate or kneedle_candidate or pareto_candidate
    cand_row = next((r for r in lang_rows if r["requested_vocab_size"] == primary_candidate), lang_rows[-1])

    return {
        "language": language,
        "candidate_vocab_size": primary_candidate,
        "threshold_candidate": th_candidate,
        "kneedle_elbow_candidate": kneedle_candidate,
        "pareto_candidate": pareto_candidate,
        "candidate_tokens_per_word": cand_row.get("tokens_per_word"),
        "candidate_zipf_exponent": cand_row.get("zipf_exponent"),
        "candidate_zipf_r2": cand_row.get("r2"),
        "candidate_js_divergence": cand_row.get("js_divergence"),
        "rationale": "; ".join(th_reasons) if th_reasons else f"Identified via geometric elbow point on compression curve (Kneedle algorithm)",
        "confidence": "High (Consensus)" if (th_candidate == kneedle_candidate and th_candidate is not None) else "Moderate (Empirical)",
    }
