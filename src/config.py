"""Configuration loader and schema validation for experiments."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "languages": ["english", "hindi", "arabic"],
    "download_wikipedia": True,
    "max_chars_demo": 15000,
    "max_chars_full": 150000,
    "train_val_split": 0.8,
    "vocab_sizes_demo": [64, 128, 256, 512, 1024],
    "vocab_sizes_full": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    "fit_min_rank": 2,
    "fit_max_rank": None,
    "stability_metric": "js_divergence",
    "thresholds": {
        "relative_efficiency_gain": 0.03,
        "js_divergence": 0.05,
        "zipf_delta": 0.05,
        "r2_delta": 0.02,
        "marginal_elasticity": 0.10,
    },
    "pretrained_tokenizers": {
        "llama": "hf-internal-testing/llama-tokenizer",
        "qwen": "Qwen/Qwen2.5-0.5B",
        "deepseek": "deepseek-ai/deepseek-llm-7b-base",
        "gpt2": "openai-community/gpt2",
    },
}


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file with fallbacks."""
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG.copy()

    content = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(data)
            return cfg
    except Exception:
        pass

    try:
        # Try JSON cleanup
        clean_json = re.sub(r",\s*([}\]])", r"\1", content)
        data = json.loads(clean_json)
        if isinstance(data, dict):
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(data)
            return cfg
    except Exception as exc:
        raise ValueError(f"Failed to parse config from {config_path}: {exc}") from exc

    return DEFAULT_CONFIG.copy()
