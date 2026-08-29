"""Frequency counting, ranking, probability distributions, and vocabulary statistics."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


def compute_frequency_distribution(tokens: List[str]) -> Tuple[List[dict], Counter]:
    """Compute sorted rank-frequency table and Counter object."""
    counts = Counter(tokens)
    total_tokens = len(tokens)
    
    ranked_data = []
    cumulative_count = 0
    
    for rank, (token, freq) in enumerate(counts.most_common(), start=1):
        cumulative_count += freq
        ranked_data.append({
            "rank": rank,
            "token": token,
            "frequency": freq,
            "probability": freq / max(1, total_tokens),
            "log_rank": math.log(rank),
            "log_frequency": math.log(freq),
            "cumulative_probability": cumulative_count / max(1, total_tokens),
        })
        
    return ranked_data, counts


def compute_shannon_entropy(counts: Counter) -> float:
    """Calculate Shannon entropy in bits for token distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for freq in counts.values():
        p = freq / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_distribution_metrics(tokens: List[str], target_vocab_size: int | None = None) -> dict:
    """Compute detailed distribution metrics including hapax legomena and entropy."""
    ranked_data, counts = compute_frequency_distribution(tokens)
    total_tokens = len(tokens)
    unique_tokens = len(counts)
    
    # Hapax legomena (tokens appearing exactly once)
    hapax_count = sum(1 for f in counts.values() if f == 1)
    hapax_ratio = hapax_count / max(1, unique_tokens)
    
    # Vocabulary utilization
    allocated_vocab = target_vocab_size if target_vocab_size else unique_tokens
    utilization = unique_tokens / max(1, allocated_vocab)
    
    frequencies = list(counts.values())
    
    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "allocated_vocab": allocated_vocab,
        "vocabulary_utilization": round(utilization, 4),
        "hapax_legomena_count": hapax_count,
        "hapax_ratio": round(hapax_ratio, 4),
        "shannon_entropy_bits": round(compute_shannon_entropy(counts), 4),
        "mean_frequency": round(float(np.mean(frequencies)), 2) if frequencies else 0.0,
        "median_frequency": round(float(np.median(frequencies)), 2) if frequencies else 0.0,
        "top_10_mass": round(sum(d["probability"] for d in ranked_data[:10]), 4) if ranked_data else 0.0,
        "top_50_mass": round(sum(d["probability"] for d in ranked_data[:50]), 4) if ranked_data else 0.0,
    }
