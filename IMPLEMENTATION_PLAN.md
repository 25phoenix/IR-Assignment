# Implementation Plan & Experimental Methodology

## 1. Goal & Research Vision
Build a complete, mathematically rigorous, and reproducible empirical pipeline to investigate Zipf's Law in subword tokenization across English, Hindi, and Arabic. The framework examines cross-linguistic disparities (the "Tokenizer Tax"), evaluates leading open LLM tokenizers (LLaMA, Qwen, DeepSeek, GPT-2, Tiktoken), conducts strictly controlled BPE vocabulary sweeps, and establishes analytical criteria for vocabulary-size "Sweet Spot" selection.

---

## 2. Theoretical Hypotheses & Loophole Analysis

1. **Hypothesis 1 (Zipf Distortion in Subwords)**:
   - *Natural Language Mechanism*: Open vocabulary + communicative efficiency $\implies f(r) \propto r^{-s}$ with $s \approx 1$.
   - *Subword Mechanism*: Greedy iterative entropy compression merges high-frequency subword pairs $\implies$ head flattening ($\beta > 0$ in Zipf-Mandelbrot) and artificial tail truncation at vocabulary budget $V$.
2. **Hypothesis 2 (Cross-Linguistic Asymmetry / Tokenizer Tax)**:
   - English (Latin alphabet, concatenative morphology, 1 byte/char) tokenizes efficiently.
   - Hindi (Devanagari script, 3 bytes/char, complex conjuncts) and Arabic (consonantal root-and-pattern morphology with attached proclitics) suffer severe sequence length inflation on English-centric tokenizers (e.g. LLaMA SentencePiece).
3. **Hypothesis 3 (Analytical Sweet Spot)**:
   - Optimal vocabulary size is an information-theoretic and compute trade-off between sequence compression ($\text{Tokens Per Word}$) and model parameters ($V \times d_{\text{model}}$).
   - An intrinsic knee/elbow point can be identified where marginal compression elasticity $\mathcal{E} < 0.10$ and Jensen-Shannon divergence $D_{\text{JS}} < 0.05$.

---

## 3. Modular Implementation Architecture

1. `src/config.py`: Configuration loading, validation, default hyperparameter specification.
2. `src/data_collection.py`: Multi-article Wikipedia text acquisition via MediaWiki API with authentic offline academic fallback corpora.
3. `src/preprocessing.py`: Unicode NFC normalization, URL/markup stripping, and script-preserving regex boundary extraction.
4. `src/tokenizer_training.py`: Fast HuggingFace Rust BPE training with full byte-level fallback for 100% UTF-8 character coverage.
5. `src/frequency_analysis.py`: Rank-frequency counting, Shannon entropy computation, empirical PMF/CDF, and hapax legomena profiling.
6. `src/zipf_analysis.py`: Linear OLS regression ($\log f = C - s \log r$), nonlinear Zipf-Mandelbrot fitting ($\log f = C - s \log(r + \beta)$), and head/tail goodness-of-fit.
7. `src/stability_analysis.py`: Jensen-Shannon divergence $D_{\text{JS}}$, Total Variation distance $\text{TV}$, Spearman rank correlation, and marginal elasticity $\mathcal{E}$.
8. `src/tokenizer_comparison.py`: Benchmarking LLaMA, Qwen, DeepSeek, GPT-2, and Tiktoken CL100K on full corpora and parallel semantic samples.
9. `src/sweet_spot.py`: Consensus sweet-spot selection (Thresholds, Kneedle Geometric Elbow, Bi-Criterion Pareto Cost).
10. `src/utils.py`: Publication-grade Matplotlib/Seaborn visualizer, CSV/JSON serialization.
11. `experiments/`: Dedicated execution modules for each research question (`language_comparison.py`, `existing_tokenizers.py`, `vocabulary_sizes.py`, `run_all.py`).
12. `report/generate_report.py`: Automated scientific report generator.

---

## 4. Empirical Validation & Verification Protocol

- [x] Static syntax and import validation across all modules.
- [x] Execution of `run_experiment.py --mode demo` verifying all 5 phases.
- [x] Execution of `run_experiment.py --mode full` on 150,000 characters of authentic Wikipedia text per language.
- [x] Generation of 10 publication-quality plots under `results/plots/`.
- [x] Generation of 6 machine-readable CSV tables under `results/tables/`.
- [x] Verification of trained BPE JSON tokenizer artifacts under `results/models/`.
- [x] Automated compilation of the scientific research report under `report/report.md`.
