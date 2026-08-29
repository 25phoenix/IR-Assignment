# Assignment 2: Zipf's Law and Multilingual Subword Tokenization

A rigorous, fully reproducible research framework and empirical investigation exploring the emergence, distortions, and stabilization of **Zipf's Law** across multilingual subword tokenization, comparing leading LLM tokenizers (LLaMA, Qwen, DeepSeek, GPT-2, Tiktoken), and evaluating information-theoretic vocabulary-size **"Sweet Spot"** stopping criteria.

---

## 1. Research Objectives & Pivotal Questions

1. **Does Zipf's Law hold for subword tokens?**
   - We assess whether BPE algorithmic optimization preserves natural open-vocabulary power-law scaling ($f(r) \propto r^{-s}$) or creates a distorted, truncated distribution with a flattened head ($\beta > 0$ in the Zipf-Mandelbrot generalization).
2. **How does language structure impact tokenization efficiency (The "Tokenizer Tax")?**
   - We quantify cross-linguistic fertility ($\text{Tokens Per Word}$) across English (Latin script), Hindi (Devanagari alphasyllabary), and Arabic (templatic root-and-clitic morphology).
3. **How do state-of-the-art LLM tokenizers compare?**
   - We benchmark **LLaMA** (32K), **Qwen** (151K), **DeepSeek** (100K), **GPT-2** (50K), and **Tiktoken CL100K** on identical texts and parallel sentences.
4. **Is there an intrinsic "Sweet Spot" for vocabulary size?**
   - We formulate and test analytical stopping criteria based on marginal fertility elasticity $\mathcal{E}$, Jensen-Shannon distributional divergence $D_{\text{JS}}$, Zipf slope convergence $|\Delta s|$, the geometric Kneedle elbow point, and bi-criterion Pareto cost.

---

## 2. Project Architecture

```
ASS2/
├── config.yaml                    # Master experiment configuration
├── requirements.txt               # Dependencies
├── README.md                      # Scientific overview and run instructions
├── IMPLEMENTATION_PLAN.md         # Detailed architectural and validation plan
├── run_experiment.py              # CLI entry point (--mode demo / --mode full)
│
├── data/
│   ├── raw/                       # Cached Wikipedia extracts (English, Hindi, Arabic)
│   └── processed/                 # Train / Validation splits (80/20)
│
├── src/
│   ├── config.py                  # Config loader and schema validation
│   ├── data_collection.py         # MediaWiki API acquisition with authentic fallbacks
│   ├── preprocessing.py           # Unicode NFC normalization and script-preserving tokenization
│   ├── tokenizer_training.py       # Fast Rust BPE via HuggingFace tokenizers + byte fallback
│   ├── frequency_analysis.py       # Rank-frequency distributions, entropy, and top-k metrics
│   ├── zipf_analysis.py            # OLS Zipf & Non-linear Zipf-Mandelbrot regression
│   ├── stability_analysis.py       # Jensen-Shannon divergence, TV distance, Spearman correlation
│   ├── tokenizer_comparison.py     # SOTA LLM tokenizer benchmarking & Tokenizer Tax analysis
│   ├── sweet_spot.py               # Analytical stopping criteria (Kneedle, Thresholds, Pareto)
│   └── utils.py                    # Publication-quality plotting and CSV/JSON exporters
│
├── experiments/
│   ├── language_comparison.py      # Experiment 1: Multilingual Word Zipf Analysis
│   ├── existing_tokenizers.py      # Experiment 2: LLM Tokenizer Benchmarking
│   ├── vocabulary_sizes.py         # Experiment 3: Controlled Vocabulary Sweeps & Sweet Spot
│   └── run_all.py                  # Master pipeline orchestrator
│
├── results/
│   ├── models/                    # Trained BPE JSON tokenizer artifacts
│   ├── tables/                    # CSV summary tables
│   ├── plots/                     # High-resolution publication plots
│   └── raw/                       # Raw frequency distributions and run metadata
│
└── report/
    ├── generate_report.py          # Dynamic report generator pulling live results
    └── report.md                   # Complete generated scientific research report
```

---

## 3. Installation & Environment Setup

Using `uv` (recommended):

```powershell
# Create virtual environment and install packages
uv venv .venv --python 3.12
uv pip install -r requirements.txt
```

Using standard `pip`:

```powershell
python -m pip install -r requirements.txt
```

---

## 4. How to Run Experiments

### Fast Demo / Smoke-Test Mode:
```powershell
.\.venv\Scripts\python.exe run_experiment.py --mode demo --config config.yaml
```
*(Runs in seconds, downloads sample Wikipedia articles, benchmarks tokenizers, trains 15 BPE models, and builds report).*

### Full-Scale Research Mode:
```powershell
.\.venv\Scripts\python.exe run_experiment.py --mode full --config config.yaml
```
*(Fetches 150,000 characters of balanced Wikipedia text per language across 16 encyclopedic topics, performs extensive BPE sweeps up to 16,384 vocab, computes all divergence metrics, and populates publication tables and report).*

---

## 5. Summary of Key Research Findings

1. **Subword Zipf Distortion**:
   - Natural words follow power-law scaling ($R^2 > 0.96$). However, BPE subwords alter the distribution: as vocabulary size $V$ grows, the head flattens ($\beta > 0$) and the tail is bounded by $|V|$.
2. **Empirical "Tokenizer Tax"**:
   - LLaMA (32K vocabulary) requires **5.84 tokens per word** on Hindi and **5.46 tokens per word** on Arabic, compared to only **1.47 tokens per word** on English (a $4\times$ penalty).
   - Qwen (151K vocabulary) mitigates this penalty significantly (**2.25 tokens per word** on Arabic and **5.04 on Hindi**).
3. **Diminishing Returns & Vocabulary Sweet Spots**:
   - For English, doubling vocabulary size beyond $V = 4096 - 8192$ yields $< 1.3\%$ marginal compression gain with $D_{\text{JS}} \approx 0.01$.
   - For Hindi, syllabic conjuncts stabilize at $V \approx 2048$.
   - For Arabic, complex templatic morphology and clitic attachments benefit from larger budgets ($V \approx 8192 - 16384$).

---

## 6. Output Artifacts

- **Report**: `report/report.md`
- **Tables**:
  - `results/tables/corpus_metadata.csv`
  - `results/tables/language_comparison.csv`
  - `results/tables/existing_tokenizers.csv`
  - `results/tables/parallel_segmentation_samples.csv`
  - `results/tables/vocabulary_experiment.csv`
  - `results/tables/sweet_spots.csv`
- **Plots**:
  - `results/plots/zipf_multilingual_comparison.png`
  - `results/plots/tokenizer_tax_fertility.png`
  - `results/plots/vocab_vs_tokens_per_word.png`
  - `results/plots/vocab_vs_zipf_exponent.png`
  - `results/plots/vocab_vs_r2.png`
  - `results/plots/vocab_vs_js_divergence.png`
  - `results/plots/vocab_vs_marginal_gain.png`
