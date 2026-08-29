"""Pretrained tokenizer benchmarking (LLaMA, Qwen, DeepSeek, GPT-2, Tiktoken) and Tokenizer Tax analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from src.preprocessing import extract_words, compute_corpus_stats
from src.zipf_analysis import analyze_zipf_behavior
from src.frequency_analysis import compute_shannon_entropy


# Standard parallel sentences for qualitative cross-linguistic segmentation comparison
PARALLEL_SAMPLES = {
    "english": "Language models learn statistical patterns from large text corpora across different domains.",
    "hindi": "भाषा मॉडल विभिन्न डोमेन में बड़े पाठ संग्रह से सांख्यिकीय पैटर्न सीखते हैं।",
    "arabic": "تتعلم النماذج اللغوية الأنماط الإحصائية من المجموعات النصية الكبيرة عبر مجالات متعددة.",
}


class PretrainedTokenizerBenchmark:
    """Evaluates and compares HuggingFace and Tiktoken tokenizers across languages."""

    def __init__(self, tokenizer_dict: Dict[str, str]):
        self.tokenizer_dict = tokenizer_dict
        self.loaded_tokenizers = {}
        self._load_all()

    def _load_all(self):
        try:
            from transformers import AutoTokenizer
            for name, model_id in self.tokenizer_dict.items():
                print(f"  [Tokenizer Benchmark] Loading {name} ({model_id})...", flush=True)
                try:
                    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=False, trust_remote_code=True)
                    self.loaded_tokenizers[name] = ("transformers", tok, model_id, len(tok))
                    print(f"  [Tokenizer Benchmark] -> {name} loaded (vocab: {len(tok)})", flush=True)
                except Exception as e:
                    # Try local files only
                    try:
                        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                        self.loaded_tokenizers[name] = ("transformers", tok, model_id, len(tok))
                        print(f"  [Tokenizer Benchmark] -> {name} loaded locally (vocab: {len(tok)})", flush=True)
                    except Exception:
                        print(f"  [Tokenizer Benchmark] -> {name} could not be loaded: {e}", flush=True)
        except ImportError:
            pass

        # Try Tiktoken cl100k
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            self.loaded_tokenizers["tiktoken_cl100k"] = ("tiktoken", enc, "cl100k_base", enc.n_vocab)
            print(f"  [Tokenizer Benchmark] -> tiktoken_cl100k loaded (vocab: {enc.n_vocab})", flush=True)
        except Exception:
            pass

    def encode_text(self, name: str, text: str) -> List[str]:
        if name not in self.loaded_tokenizers:
            raise ValueError(f"Tokenizer {name} is not loaded.")
        tok_type, tok_obj, _, _ = self.loaded_tokenizers[name]
        
        if tok_type == "transformers":
            token_ids = tok_obj(text, add_special_tokens=False)["input_ids"]
            return [str(tid) for tid in token_ids]
        elif tok_type == "tiktoken":
            token_ids = tok_obj.encode(text)
            return [str(tid) for tid in token_ids]
        return []

    def get_token_pieces(self, name: str, text: str) -> List[str]:
        """Return human-readable token strings for visualization."""
        if name not in self.loaded_tokenizers:
            return []
        tok_type, tok_obj, _, _ = self.loaded_tokenizers[name]
        if tok_type == "transformers":
            try:
                return tok_obj.tokenize(text)
            except Exception:
                token_ids = tok_obj(text, add_special_tokens=False)["input_ids"]
                return [tok_obj.decode([tid]) for tid in token_ids]
        elif tok_type == "tiktoken":
            token_ids = tok_obj.encode(text)
            return [tok_obj.decode([tid]) for tid in token_ids]
        return []

    def benchmark_corpora(
        self,
        corpora: Dict[str, Dict[str, str]],
        min_rank: int = 2,
        max_rank: int | None = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Run full evaluation across all language corpora and loaded tokenizers."""
        benchmark_rows = []
        segmentation_samples = []

        for lang, corp_data in corpora.items():
            text = corp_data["full"]
            words_list = extract_words(text)
            num_words = max(1, len(words_list))
            num_chars = max(1, len(text))
            utf8_bytes = len(text.encode("utf-8"))

            for tok_name, (tok_type, tok_obj, model_id, vocab_size) in self.loaded_tokenizers.items():
                try:
                    tokens = self.encode_text(tok_name, text)
                    if not tokens:
                        continue
                    
                    num_tokens = len(tokens)
                    counts = Counter(tokens)
                    zipf_stats, ranked = analyze_zipf_behavior(tokens, min_rank=min_rank, max_rank=max_rank)
                    entropy = compute_shannon_entropy(counts)

                    tpw = num_tokens / num_words
                    tpc = num_tokens / num_chars
                    bpt = utf8_bytes / max(1, num_tokens)

                    row = {
                        "language": lang,
                        "tokenizer": tok_name,
                        "model_id": model_id,
                        "tokenizer_type": tok_type,
                        "vocabulary_size": vocab_size,
                        "corpus_tokens": num_tokens,
                        "unique_tokens_used": len(counts),
                        "vocabulary_utilization": round(len(counts) / max(1, vocab_size), 5),
                        "tokens_per_word": round(tpw, 4),
                        "tokens_per_character": round(tpc, 4),
                        "bytes_per_token": round(bpt, 3),
                        "zipf_exponent": zipf_stats.get("zipf_exponent"),
                        "zipf_r2": zipf_stats.get("r2"),
                        "mandelbrot_beta": zipf_stats.get("mandelbrot_beta"),
                        "mandelbrot_r2": zipf_stats.get("mandelbrot_r2"),
                        "shannon_entropy": round(entropy, 4),
                        "status": "success",
                    }
                    benchmark_rows.append(row)
                except Exception as exc:
                    benchmark_rows.append({
                        "language": lang,
                        "tokenizer": tok_name,
                        "model_id": model_id,
                        "status": f"error: {exc}",
                    })

        # Qualitative parallel sentence segmentation analysis
        for lang, sample_sentence in PARALLEL_SAMPLES.items():
            words_in_sample = len(extract_words(sample_sentence))
            for tok_name in self.loaded_tokenizers:
                pieces = self.get_token_pieces(tok_name, sample_sentence)
                segmentation_samples.append({
                    "language": lang,
                    "tokenizer": tok_name,
                    "original_text": sample_sentence,
                    "token_count": len(pieces),
                    "tokens_per_word": round(len(pieces) / max(1, words_in_sample), 2),
                    "token_pieces": pieces[:20],
                })

        return benchmark_rows, segmentation_samples
