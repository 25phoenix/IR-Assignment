# Implementation Plan

## Goal

Build a reproducible English/Hindi/Arabic tokenization experiment that measures
rank-frequency Zipf behavior, compares optional pretrained tokenizers, evaluates
controlled BPE vocabulary sizes, and selects a candidate stopping point from
measured diminishing returns and distributional stability.

## Design

1. `run_experiment.py` is the single entry point for `demo` and `full` modes.
2. `src/pipeline.py` orchestrates corpus preparation, tokenization, frequency
   analysis, plotting, stability, sweet-spot selection, and report generation.
3. The corpus uses deterministic built-in multilingual seed text in demo mode;
   full mode can download comparable Wikipedia REST extracts with a documented
   fallback when network access is unavailable.
4. A small deterministic byte/character BPE implementation keeps the project
   runnable without model weights or optional packages. Existing LLaMA, Qwen,
   and Kimi tokenizers are attempted only when `transformers` is installed and
   model access is available; failures are recorded, never hidden.
5. All generated CSV, JSON, PNG, tokenizer artifacts, and the data-backed
   report are written below `results/`, `data/`, and `report/`.

## Validation

- Compile every Python module.
- Run `python run_experiment.py --mode demo --config config.yaml`.
- Verify non-empty CSV/JSON/PNG outputs and the generated report.

## Limitations

Demo data is a compact deterministic smoke-test corpus, not a substitute for
Wikipedia. Full mode is intentionally configurable and records source,
download status, and corpus statistics so conclusions remain traceable.
