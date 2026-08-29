"""Controlled BPE Tokenizer training and encoding with Byte-level coverage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

from src.preprocessing import extract_words


class FastBPETokenizer:
    """Wrapper around HuggingFace Tokenizers BPE model with byte-level fallback."""

    def __init__(self, vocab_size: int, language: str):
        self.vocab_size = vocab_size
        self.language = language
        self.tokenizer = None
        self._is_trained = False

    def train_from_text(self, text: str, min_frequency: int = 2) -> FastBPETokenizer:
        try:
            from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
            
            # Initialize BPE model with byte fallback
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_frequency,
                special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                show_progress=False,
            )
            
            # Train from in-memory text iterator
            tokenizer.train_from_iterator([text], trainer=trainer)
            self.tokenizer = tokenizer
            self._is_trained = True
            return self
        except Exception:
            # Fallback to Pure Python BPE
            return self._train_fallback_bpe(text)

    def _train_fallback_bpe(self, text: str) -> FastBPETokenizer:
        """Pure-Python character BPE fallback."""
        words_list = extract_words(text)
        sequences = [list(w) + ["</w>"] for w in words_list]
        base_vocab = sorted({char for seq in sequences for char in seq})
        merges = []
        
        while len(base_vocab) + len(merges) < self.vocab_size:
            pair_counts = Counter(
                (seq[i], seq[i + 1])
                for seq in sequences
                for i in range(len(seq) - 1)
            )
            if not pair_counts:
                break
            best_pair, freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break
            merged = best_pair[0] + best_pair[1]
            merges.append((best_pair[0], best_pair[1], merged))
            base_vocab.append(merged)
            
            new_sequences = []
            for seq in sequences:
                res, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best_pair:
                        res.append(merged)
                        i += 2
                    else:
                        res.append(seq[i])
                        i += 1
                new_sequences.append(res)
            sequences = new_sequences
            
        self._fallback_model = {
            "vocab": base_vocab,
            "merges": merges,
            "vocab_size": len(base_vocab),
        }
        self._is_trained = True
        return self

    def encode(self, text: str) -> List[str]:
        """Tokenize text into a sequence of token strings."""
        if not self._is_trained:
            raise RuntimeError("Tokenizer has not been trained yet.")
            
        if self.tokenizer is not None:
            encoding = self.tokenizer.encode(text)
            return encoding.tokens
        else:
            # Fallback encode
            tokens = []
            for word in extract_words(text):
                seq = list(word) + ["</w>"]
                for left, right, merged in self._fallback_model["merges"]:
                    res, i = [], 0
                    while i < len(seq):
                        if i + 1 < len(seq) and seq[i] == left and seq[i + 1] == right:
                            res.append(merged)
                            i += 2
                        else:
                            res.append(seq[i])
                            i += 1
                    seq = res
                tokens.extend(seq)
            return tokens

    def get_actual_vocab_size(self) -> int:
        if self.tokenizer is not None:
            return self.tokenizer.get_vocab_size()
        elif hasattr(self, "_fallback_model"):
            return len(self._fallback_model["vocab"])
        return self.vocab_size

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.tokenizer is not None:
            self.tokenizer.save(str(path))
        else:
            path.write_text(json.dumps(self._fallback_model, ensure_ascii=False, indent=2), encoding="utf-8")


def train_controlled_bpe(
    text: str,
    target_vocab_size: int,
    language: str,
    save_dir: Optional[Path] = None,
) -> FastBPETokenizer:
    """Train a BPE tokenizer with a strictly controlled vocabulary budget."""
    tok = FastBPETokenizer(vocab_size=target_vocab_size, language=language)
    tok.train_from_text(text)
    if save_dir:
        save_path = save_dir / f"{language}_bpe_{target_vocab_size}.json"
        tok.save(save_path)
    return tok
