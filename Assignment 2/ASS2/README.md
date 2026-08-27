# Assignment 2: Zipf's Law and Tokenization

A reproducible multilingual research pipeline for studying Zipf rank-frequency behavior, tokenizer differences, controlled BPE vocabulary sizes, distributional stability, and an experimental vocabulary stopping criterion.

## Research Design

English, Hindi, and Arabic use the same normalization and Unicode-aware word extraction. The controlled experiment fixes the corpus, preprocessing, and a deterministic character-level BPE algorithm while varying only requested vocabulary size. Zipf statistics fit `log(frequency)` against `log(rank)` from rank 2 onward. Consecutive token distributions are compared with Jensen-Shannon divergence. A candidate vocabulary is the first size passing configurable thresholds for efficiency gain, JS divergence, Zipf exponent change, and R2 change.

This criterion is a measurable heuristic, not a universal optimum. Demo mode is a smoke test; use balanced Wikipedia data for research claims.

## Install and Run

```powershell
python -m pip install -r requirements.txt
python run_experiment.py --mode demo --config config.yaml
python run_experiment.py --mode full --config config.yaml
```

Demo mode requires no downloads or model weights. Full mode automatically reads local UTF-8 files named `corpus.txt` under each language directory; when they are absent it uses the deterministic fallback corpus. `max_chars_full` controls the maximum characters used.

## Use Real Wikipedia Data

Download or export comparable plain-text Wikipedia extracts, then place them at:

```text
data/raw/english/corpus.txt
data/raw/hindi/corpus.txt
data/raw/arabic/corpus.txt
```

Keep the same approximate number of characters or documents for each language. The runner reads these files only in `full` mode, applies the same preprocessing, truncates each to `max_chars_full`, and records the local path in `results/tables/corpus_metadata.csv`.

For a quick online test, set `download_full_wikipedia` to `true` in `config.yaml`. That option retrieves a small set of Wikimedia REST article summaries, not a full Wikipedia dump. For serious experiments, use Wikimedia dumps or a reproducible Wikipedia text extract and document the download date and filtering procedure.

## Optional Existing Tokenizers

Install `transformers` separately if desired:

```powershell
python -m pip install transformers
python run_experiment.py --mode demo --config config.yaml
```

The runner uses `local_files_only=True`, so it never silently downloads large model weights. LLaMA, Qwen, or Kimi rows are emitted only when their tokenizer files are already available locally. Missing access is recorded in `results/raw/run_metadata.json` and `report/report.md`.

## Outputs

- `results/tables/corpus_metadata.csv`: corpus sizes and provenance
- `results/tables/vocabulary_experiment.csv`: language x vocabulary metrics and deltas
- `results/tables/existing_tokenizers.csv`: optional pretrained tokenizer measurements
- `results/tables/sweet_spots.csv`: data-derived candidate sizes
- `results/plots/`: Zipf and vocabulary-size PNG plots
- `results/raw/`: ranked frequencies and run metadata
- `report/report.md`: generated report based only on produced rows
- `data/raw/<language>/`: exact prepared text used by each run

## Reproducibility and Limitations

`config.yaml` contains languages, seed, vocabulary sizes, fitting settings, and thresholds. Runs are deterministic for the built-in corpus. Results depend on corpus balance, Unicode segmentation, fit range, sample size, BPE implementation, and the chosen stability thresholds. The report marks unavailable pretrained tokenizers and does not fabricate results.
