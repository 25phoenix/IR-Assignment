"""Unicode-aware text preprocessing and normalization pipeline."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# Regex to remove HTML tags and wiki artifacts
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WIKI_REF_RE = re.compile(r"\[\d+\]|\[edit\]|\[citation needed\]", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

# Unicode-aware word token regex covering Latin, Devanagari, Arabic, and numbers
WORD_TOKEN_RE = re.compile(r"[\w\u0900-\u097F\u0600-\u06FF]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Apply standard NFC normalization, strip HTML/URLs, and normalize whitespace."""
    if not text:
        return ""
    # NFC Unicode Normalization ensures canonical decomposition followed by canonical composition
    text = unicodedata.normalize("NFC", text)
    # Strip HTML tags
    text = HTML_TAG_RE.sub(" ", text)
    # Replace URLs with token
    text = URL_RE.sub(" ", text)
    # Remove Wiki citation marks like [1], [edit]
    text = WIKI_REF_RE.sub(" ", text)
    # Normalize multiple whitespace characters to a single space
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_words(text: str) -> List[str]:
    """Extract individual words using Unicode-aware word boundaries without destroying scripts."""
    normalized = normalize_text(text)
    return WORD_TOKEN_RE.findall(normalized)


def compute_corpus_stats(text: str) -> dict:
    """Compute character, word, byte, and unique word statistics."""
    words = extract_words(text)
    utf8_bytes = len(text.encode("utf-8"))
    char_count = len(text)
    word_count = len(words)
    unique_words = len(set(words))
    
    return {
        "character_count": char_count,
        "word_count": word_count,
        "byte_count": utf8_bytes,
        "unique_words": unique_words,
        "bytes_per_char": utf8_bytes / max(1, char_count),
        "chars_per_word": char_count / max(1, word_count),
    }


def split_train_val(text: str, train_ratio: float = 0.8) -> Tuple[str, str]:
    """Split text corpus into train and validation portions by word boundaries."""
    words = extract_words(text)
    split_idx = int(len(words) * train_ratio)
    train_words = words[:split_idx]
    val_words = words[split_idx:]
    return " ".join(train_words), " ".join(val_words)
