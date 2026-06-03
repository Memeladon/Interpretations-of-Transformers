"""Очистка, нормализация, дедупликация, фильтрация, балансировка"""

from __future__ import annotations

import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from src.data_schema import PREPROCESSING_VERSION_DEFAULT, normalize_text_key

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_records(
    records: list[dict[str, Any]],
    *,
    version: str = PREPROCESSING_VERSION_DEFAULT,
    clean: bool = True,
    normalize_ws: bool = True,
    deduplicate: bool = True,
    filter_empty: bool = True,
    max_text_length: int | None = 10000,
    balance_classes: bool = False,
    balance_max_per_class: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in records:
        text = str(row.get("text", ""))
        if clean:
            text = clean_text(text)
        if normalize_ws:
            text = normalize_whitespace(text)
        if filter_empty and not text:
            continue
        if max_text_length and len(text) > max_text_length:
            text = text[:max_text_length]

        key = normalize_text_key(text)
        if deduplicate and key in seen:
            continue
        seen.add(key)

        row = {**row, "text": text}
        meta = dict(row.get("metadata") or {})
        meta["preprocessing_version"] = version
        meta["text_length"] = len(text)
        row["metadata"] = meta
        out.append(row)

    if balance_classes:
        out = _balance_by_semantic_label(out, balance_max_per_class, seed)

    logger.info("preprocess: %s → %s records", len(records), len(out))
    return out


def _balance_by_semantic_label(
    records: list[dict[str, Any]],
    max_per_class: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Балансировка по y_semantic (single-label)."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        label = row.get("y_semantic", row.get("label"))
        if label is None:
            by_label["__none__"].append(row)
            continue
        by_label[str(label)].append(row)

    if not by_label or len(by_label) <= 1:
        return records

    counts = {k: len(v) for k, v in by_label.items() if k != "__none__"}
    if not counts:
        return records
    target = min(counts.values()) if max_per_class is None else min(min(counts.values()), max_per_class)

    balanced: list[dict[str, Any]] = []
    for label, group in by_label.items():
        if label == "__none__":
            balanced.extend(group)
            continue
        indices = list(range(len(group)))
        rng.shuffle(indices)
        take = min(target, len(group))
        balanced.extend(group[i] for i in sorted(indices[:take]))
    rng.shuffle(balanced)
    logger.info("balance_classes: target_per_class=%s, total=%s", target, len(balanced))
    return balanced
