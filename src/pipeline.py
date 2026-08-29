"""Backwards-compatible pipeline wrapper forwarding to modular experiments."""
from __future__ import annotations

from src.config import load_config
from experiments.run_all import run_pipeline


def run(mode: str, config: dict):
    results = run_pipeline(mode=mode)
    return (
        results["corpus_metadata"],
        results["vocabulary_sweep"],
        results["sweet_spots"],
        [],
    )
