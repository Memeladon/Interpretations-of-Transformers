"""Поверхностные признаки для leakage baselines"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

PUNCT_RE = re.compile(r"[.,!?;:—\-\"'«»()\[\]{}…]")
SPECIAL_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def _char_ngram_counts(text: str, ns: tuple[int, ...] = (1, 2, 3), dim: int = 256) -> np.ndarray:
    """Хэш char-ngram в фиксированный вектор (детерминированно)."""
    vec = np.zeros(dim, dtype=np.float32)
    t = text.lower()
    for n in ns:
        for i in range(max(0, len(t) - n + 1)):
            gram = t[i : i + n]
            vec[hash(gram) % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class SurfaceFeatureExtractor:
    def __init__(
        self,
        *,
        char_ngram_dim: int = 256,
        domain_vocabulary: list[str] | None = None,
        tokenizer=None,
    ):
        self.char_ngram_dim = char_ngram_dim
        self.domain_vocabulary = [w.lower() for w in (domain_vocabulary or [])]
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return [str(t) for t in ids]
        return WORD_RE.findall(text.lower())

    def extract_one(self, text: str, *, text_pair: str | None = None) -> dict[str, Any]:
        n = len(text)
        punct = len(PUNCT_RE.findall(text))
        special = len(SPECIAL_RE.findall(text))
        tokens = self._tokenize(text)
        freq = Counter(tokens)
        top_freq = float(max(freq.values()) / max(len(tokens), 1)) if freq else 0.0
        uniq_ratio = float(len(freq) / max(len(tokens), 1)) if tokens else 0.0

        domain_hits = 0
        if self.domain_vocabulary and text:
            low = text.lower()
            domain_hits = sum(1 for w in self.domain_vocabulary if w in low)

        overlap = 0.0
        if text_pair:
            ta = set(self._tokenize(text))
            tb = set(self._tokenize(text_pair))
            if ta or tb:
                overlap = len(ta & tb) / len(ta | tb)

        return {
            "text_length": n,
            "punctuation_density": punct / max(n, 1),
            "special_char_count": special,
            "token_count": len(tokens),
            "token_top_frequency": top_freq,
            "token_unique_ratio": uniq_ratio,
            "lexical_overlap": overlap,
            "domain_vocab_hits": domain_hits,
        }

    def extract_vector(self, text: str, *, text_pair: str | None = None) -> np.ndarray:
        scalars = self.extract_one(text, text_pair=text_pair)
        base = np.array(
            [
                scalars["text_length"],
                scalars["punctuation_density"],
                scalars["special_char_count"],
                scalars["token_count"],
                scalars["token_top_frequency"],
                scalars["token_unique_ratio"],
                scalars["lexical_overlap"],
                scalars["domain_vocab_hits"],
            ],
            dtype=np.float32,
        )
        ngrams = _char_ngram_counts(text, dim=self.char_ngram_dim)
        return np.concatenate([base, ngrams], axis=0)

    def extract_batch(self, records: list[dict[str, Any]]) -> np.ndarray:
        rows = []
        for r in records:
            pair = r.get("text_pair")
            rows.append(self.extract_vector(r["text"], text_pair=pair))
        return np.stack(rows, axis=0)


def extract_surface_matrix(
    records: list[dict[str, Any]],
    *,
    domain_vocabulary: list[str] | None = None,
    char_ngram_dim: int = 256,
) -> np.ndarray:
    return SurfaceFeatureExtractor(
        domain_vocabulary=domain_vocabulary,
        char_ngram_dim=char_ngram_dim,
    ).extract_batch(records)
