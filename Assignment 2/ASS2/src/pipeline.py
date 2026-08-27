"""Reproducible Zipf/tokenizer experiment with no mandatory model downloads."""
from __future__ import annotations
import csv, json, math, re, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG_TEXT = {
    "english": "Language models learn patterns from text. Zipf law describes how common words become rare as rank increases. This small corpus contains science, language, data, and reproducible experiments. ",
    "hindi": "भाषा मॉडल पाठ से पैटर्न सीखते हैं। जिप्फ़ का नियम बताता है कि रैंक बढ़ने पर सामान्य शब्द दुर्लभ होते जाते हैं। यह छोटा संग्रह विज्ञान, भाषा, डेटा और प्रयोगों का वर्णन करता है। ",
    "arabic": "تتعلم نماذج اللغة الأنماط من النص. يصف قانون زيبف كيف تصبح الكلمات الشائعة نادرة مع زيادة الرتبة. يحتوي هذا النص على العلم واللغة والبيانات والتجارب القابلة لإعادة الإنتاج. ",
}
WIKI_PAGES = {
    "english": ["Language_model", "Zipf%27s_law", "Tokenization"],
    "hindi": ["भाषा_मॉडल", "जिप्फ़_का_नियम", "प्राकृतिक_भाषा_प्रसंस्करण"],
    "arabic": ["نموذج_لغة", "قانون_زيبف", "معالجة_اللغات_الطبيعية"],
}
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def load_config(path: str) -> dict:
    # Accept the JSON-compatible YAML format even when a formatter leaves trailing commas.
    content = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content = re.sub(r",\s*([}\]])", r"\1", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid configuration in {path}: {exc}") from exc


def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_corpus(mode: str, config: dict) -> dict[str, str]:
    limit = config["max_chars_demo"] if mode == "demo" else config["max_chars_full"]
    corpora = {}
    raw_dir = Path("data/raw")
    for language in config["languages"]:
        corpus_file = raw_dir / language / "corpus.txt"
        if mode == "full" and corpus_file.exists():
            text = corpus_file.read_text(encoding="utf-8")
            source = f"local corpus: {corpus_file.as_posix()}"
        else:
            text = LANG_TEXT[language] * max(1, math.ceil(limit / len(LANG_TEXT[language])))
            source = "built-in deterministic corpus"
        if mode == "full" and not corpus_file.exists() and config.get("download_full_wikipedia", False):
            try:
                import requests
                pieces = []
                for title in WIKI_PAGES[language]:
                    url = f"https://{language[:2]}.wikipedia.org/api/rest_v1/page/summary/{title}"
                    response = requests.get(url, timeout=20)
                    response.raise_for_status()
                    extract = response.json().get("extract", "")
                    if extract: pieces.append(extract)
                if pieces:
                    text, source = " ".join(pieces), "Wikimedia REST page summaries"
            except Exception:
                pass
        text = preprocess(text[:limit])
        corpora[language] = text
        out = raw_dir / language
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{mode}.txt").write_text(text, encoding="utf-8")
        (out / f"{mode}_source.txt").write_text(source, encoding="utf-8")
    return corpora


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def train_bpe(text: str, target_vocab: int) -> dict:
    if target_vocab < 2:
        raise ValueError("vocabulary size must be at least 2")
    sequences = [list(w) + ["</w>"] for w in words(text)]
    base = sorted({symbol for seq in sequences for symbol in seq})
    merges = []
    while len(base) + len(merges) < target_vocab:
        pairs = Counter((seq[i], seq[i + 1]) for seq in sequences for i in range(len(seq) - 1))
        if not pairs:
            break
        pair, count = pairs.most_common(1)[0]
        if count < 2:
            break
        merged = pair[0] + pair[1]
        merges.append([pair[0], pair[1], merged, count])
        base.append(merged)
        new_sequences = []
        for seq in sequences:
            result, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                    result.append(merged); i += 2
                else:
                    result.append(seq[i]); i += 1
            new_sequences.append(result)
        sequences = new_sequences
    return {"vocab": base, "merges": merges, "requested_vocab": target_vocab}


def encode_word(word: str, model: dict) -> list[str]:
    seq = list(word) + ["</w>"]
    for left, right, merged, _ in model["merges"]:
        result, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                result.append(merged); i += 2
            else:
                result.append(seq[i]); i += 1
        seq = result
    return seq


def tokenize(text: str, model: dict) -> list[str]:
    return [token for word in words(text) for token in encode_word(word, model)]


def zipf_stats(tokens: list[str], min_rank: int = 2, max_rank: int | None = None) -> tuple[dict, list[tuple[int, int, str]]]:
    ranked = [(rank, freq, token) for rank, (token, freq) in enumerate(Counter(tokens).most_common(), 1)]
    selected = ranked[min_rank - 1:max_rank]
    if len(selected) < 2:
        return {"zipf_exponent": None, "intercept": None, "r2": None, "sample_size": len(selected), "fit_min_rank": min_rank, "fit_max_rank": max_rank}, ranked
    x = np.log([r for r, _, _ in selected]); y = np.log([f for _, f, _ in selected])
    slope, intercept = np.polyfit(x, y, 1); predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"zipf_exponent": float(-slope), "intercept": float(intercept), "r2": float(1 - ss_res / ss_tot) if ss_tot else 1.0, "sample_size": len(selected), "fit_min_rank": min_rank, "fit_max_rank": max_rank}, ranked


def js_divergence(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b); total_a, total_b = sum(a.values()), sum(b.values())
    p = np.array([a[k] / total_a for k in keys], dtype=float); q = np.array([b[k] / total_b for k in keys], dtype=float); m = (p + q) / 2
    p_term = p[p > 0] * np.log2(p[p > 0] / m[p > 0])
    q_term = q[q > 0] * np.log2(q[q > 0] / m[q > 0])
    return float(0.5 * p_term.sum() + 0.5 * q_term.sum())


def csv_write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_lines(path: Path, rows: list[dict], x: str, y: str, group: str, xlabel: str, ylabel: str, logx=False):
    plt.figure(figsize=(8, 5))
    for name in sorted({str(r[group]) for r in rows}):
        subset = [r for r in rows if str(r[group]) == name and r.get(y) is not None]
        if subset: plt.plot([r[x] for r in subset], [r[y] for r in subset], marker="o", label=name)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend();
    if logx: plt.xscale("log")
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(path, dpi=160); plt.close()


def optional_tokenizers(texts: dict[str, str], config: dict) -> tuple[list[dict], list[str]]:
    try:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
    except ImportError:
        return [], ["transformers is not installed; pretrained tokenizer comparison was skipped."]
    rows, errors = [], []
    for name, model_id in config["pretrained_tokenizers"].items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            for language, text in texts.items():
                encoded = tok(text, add_special_tokens=False)["input_ids"]
                stats, _ = zipf_stats([str(item) for item in encoded])
                rows.append({"language": language, "tokenizer": name, "model_id": model_id, "vocabulary_size": len(tok), "corpus_tokens": len(encoded), "unique_tokens_used": len(set(encoded)), "tokens_per_word": len(encoded) / max(1, len(words(text))), "tokens_per_character": len(encoded) / max(1, len(text)), "zipf_exponent": stats["zipf_exponent"], "r2": stats["r2"], "status": "ran"})
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return rows, errors


def generate_report(mode: str, corpus_rows: list[dict], vocab_rows: list[dict], sweet_rows: list[dict], tokenizer_rows: list[dict], errors: list[str]):
    def table(rows):
        if not rows: return "No rows generated."
        cols = list(rows[0]); return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"] + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows])
    dataset_note = "deterministic built-in seed corpus" if mode == "full" else "deterministic built-in demo corpus"
    text = f"""# Assignment 2: Zipf's Law and Tokenization

## 1. Objective
Measure rank-frequency behavior across English, Hindi, and Arabic and test whether controlled BPE vocabulary growth has diminishing returns.

## 2. Research Questions
Language, tokenizer, and vocabulary size are varied in separate experiments. The stopping rule is an experimental heuristic, not a universal law.

## 3. Dataset
Mode: `{mode}`. Text source used: {dataset_note}. Corpus metadata:

{table(corpus_rows)}

## 4. Preprocessing
NFC Unicode normalization, URL replacement, whitespace normalization, and Unicode-aware word extraction. Casing and punctuation are not blindly removed.

## 5. Zipf's Law Methodology
Token frequencies are sorted by descending count. Ordinary least squares fits `log(frequency)` against `log(rank)` from rank 2 through the configured range; head and tail deviations remain possible.

## 6. Language Comparison
{table(corpus_rows)}

## 7. Existing Tokenizer Comparison
{table(tokenizer_rows)}

## 8. Vocabulary-Size Experiment
Fixed corpus, preprocessing, and character BPE algorithm; only requested vocabulary size changes.

{table(vocab_rows)}

## 9. Stability Analysis
Jensen-Shannon divergence compares normalized token-frequency distributions between successive vocabulary sizes. It is symmetric and bounded, but no single stability metric is universally correct.

## 10. Sweet-Spot Criterion
A candidate is the first size after which relative tokens-per-word improvement, JS divergence, Zipf-exponent change, and R² change are all below configurable thresholds. The result is evidence from this corpus, not a proven optimum.

## 11. Results
{table(sweet_rows)}

## 12. Cross-Language Comparison
Compare candidates only after rerunning full mode on balanced held-out corpora; demo values are smoke-test outputs.

## 13. Limitations
The demo corpus is small and repetitive. Optional LLaMA/Qwen/Kimi models may be unavailable offline. Pretrained tokenizer rows are reported only when actually loaded.

## 14. Conclusions
No conclusion is asserted beyond the measured rows above. Reproducibility metadata and raw tables are stored under `results/`.

## 15. Future Work
Use balanced Wikipedia extracts, multiple held-out samples, bootstrap uncertainty intervals, and accessible exact Kimi tokenizer configuration.

## Execution Notes
""" + ("\n".join(f"- {e}" for e in errors) if errors else "- No execution errors recorded.") + "\n"
    Path("report/report.md").write_text(text, encoding="utf-8")


def run(mode: str, config: dict):
    np.random.seed(config["seed"]); texts = get_corpus(mode, config); out = Path("results"); (out / "raw").mkdir(parents=True, exist_ok=True)
    corpus_rows, vocab_rows, sweet_rows = [], [], []
    sizes = config["vocab_sizes_demo"] if mode == "demo" else config["vocab_sizes_full"]
    for language, text in texts.items():
        ws = words(text); source_file = Path("data/raw") / language / f"{mode}_source.txt"; source = source_file.read_text(encoding="utf-8")
        corpus_rows.append({"language": language, "source": source, "documents": 1, "characters": len(text), "words": len(ws), "raw_size_bytes": len(text.encode()), "processed_size_bytes": len(text.encode())})
        base_model = train_bpe(text, max(2, min(sizes)))
        base_tokens = tokenize(text, base_model); stats, ranked = zipf_stats(base_tokens, config["fit_min_rank"], config["fit_max_rank"])
        with (out / "raw" / f"{language}_zipf.json").open("w", encoding="utf-8") as f: json.dump({"language": language, "tokens": len(base_tokens), "ranked": ranked, "stats": stats}, f, ensure_ascii=False, indent=2)
        plt.figure(figsize=(7, 5)); plt.loglog([x[0] for x in ranked], [x[1] for x in ranked]); plt.xlabel("Rank"); plt.ylabel("Frequency"); plt.title(f"Zipf plot: {language}"); plt.tight_layout(); (out / "plots").mkdir(parents=True, exist_ok=True); plt.savefig(out / "plots" / f"zipf_{language}.png", dpi=160); plt.close()
        previous = None
        for size in sizes:
            model = train_bpe(text, size); tokens = tokenize(text, model); st, ranked = zipf_stats(tokens, config["fit_min_rank"], config["fit_max_rank"]); efficiency = len(tokens) / max(1, len(ws)); counts = Counter(tokens)
            row = {"language": language, "tokenizer": "controlled_char_bpe", "requested_vocab_size": size, "actual_vocab_size": len(model["vocab"]), "corpus_tokens": len(tokens), "unique_tokens_used": len(counts), "tokens_per_word": efficiency, "tokens_per_character": len(tokens) / max(1, len(text)), "average_token_length": sum(len(t.replace("</w>", "")) for t in tokens) / max(1, len(tokens)), "zipf_exponent": st["zipf_exponent"], "r2": st["r2"], "fit_min_rank": st["fit_min_rank"], "fit_max_rank": st["fit_max_rank"], "js_divergence": js_divergence(previous, counts) if previous else None, "relative_efficiency_gain": (previous_eff - efficiency) / max(previous_eff, 1e-12) if previous else None}
            vocab_rows.append(row); previous, previous_eff = counts, efficiency
        language_rows = [r for r in vocab_rows if r["language"] == language]
        thresholds = config["thresholds"]; candidate = None
        for i, row in enumerate(language_rows[1:], 1):
            prior = language_rows[i - 1]; checks = [row["relative_efficiency_gain"] is not None and abs(row["relative_efficiency_gain"]) < thresholds["relative_efficiency_gain"], row["js_divergence"] is not None and row["js_divergence"] < thresholds["js_divergence"], abs((row["zipf_exponent"] or 0) - (prior["zipf_exponent"] or 0)) < thresholds["zipf_delta"], abs((row["r2"] or 0) - (prior["r2"] or 0)) < thresholds["r2_delta"]]
            if all(checks): candidate = row; break
        sweet_rows.append({"language": language, "candidate_vocab_size": candidate["requested_vocab_size"] if candidate else None, "evidence": "all configured diminishing-return checks passed" if candidate else "no size passed all checks", "confidence": "experimental"})
    csv_write(out / "tables" / "corpus_metadata.csv", corpus_rows); csv_write(out / "tables" / "vocabulary_experiment.csv", vocab_rows); csv_write(out / "tables" / "sweet_spots.csv", sweet_rows)
    plot_lines(out / "plots" / "vocab_vs_tokens_per_word.png", vocab_rows, "requested_vocab_size", "tokens_per_word", "language", "Vocabulary size", "Tokens per word", True)
    plot_lines(out / "plots" / "vocab_vs_zipf_exponent.png", vocab_rows, "requested_vocab_size", "zipf_exponent", "language", "Vocabulary size", "Zipf exponent", True)
    plot_lines(out / "plots" / "vocab_vs_r2.png", vocab_rows, "requested_vocab_size", "r2", "language", "Vocabulary size", "R²", True)
    tokenizer_rows, tokenizer_errors = optional_tokenizers(texts, config); csv_write(out / "tables" / "existing_tokenizers.csv", tokenizer_rows); errors = tokenizer_errors
    metadata = {"mode": mode, "config": config, "languages": list(texts), "optional_tokenizer_errors": errors}; (out / "raw" / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_report(mode, corpus_rows, vocab_rows, sweet_rows, tokenizer_rows, errors)
    return corpus_rows, vocab_rows, sweet_rows, errors
