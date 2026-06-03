"""Единая схема: text, y_style, y_semantic, metadata"""

from __future__ import annotations

import re
from typing import Any

METADATA_KEYS = (
    "source",
    "domain",
    "language",
    "split",
    "pair_id",
    "contrast_type",
    "preprocessing_version",
    "text_length",
)

PREPROCESSING_VERSION_DEFAULT = "1.0"


def empty_metadata(
    *,
    source: str = "",
    domain: str = "",
    language: str = "ru",
    split: str = "train",
    pair_id: str | None = None,
    contrast_type: str | None = None,
    preprocessing_version: str = PREPROCESSING_VERSION_DEFAULT,
    text_length: int = 0,
) -> dict[str, Any]:
    return {
        "source": source,
        "domain": domain,
        "language": language,
        "split": split,
        "pair_id": pair_id,
        "contrast_type": contrast_type,
        "preprocessing_version": preprocessing_version,
        "text_length": text_length,
    }


def make_example(
    *,
    text: str,
    y_style: Any = None,
    y_semantic: Any = None,
    metadata: dict[str, Any] | None = None,
    legacy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = metadata or empty_metadata()
    if "text_length" not in meta or meta["text_length"] == 0:
        meta = {**meta, "text_length": len(text)}
    row: dict[str, Any] = {
        "text": text,
        "y_style": y_style,
        "y_semantic": y_semantic,
        "metadata": meta,
    }
    if legacy:
        row.update(legacy)
    return row


def legacy_from_track_mapper(legacy_row: dict[str, Any]) -> dict[str, Any]:
    """Преобразование legacy-записи datasets_load → схема ТЗ."""
    track = legacy_row.get("track", "")
    task = legacy_row.get("task_name", "")
    y_style = None
    y_semantic = None

    if track == "style":
        y_style = legacy_row.get("labels") if legacy_row.get("label_type") == "multi_label" else legacy_row.get("label")
    elif track == "tone":
        y_semantic = legacy_row.get("label")
    elif track == "semantic":
        if legacy_row.get("task_type") == "regression":
            y_semantic = legacy_row.get("label")
        else:
            y_semantic = legacy_row.get("label")

    pair_id = None
    contrast_type = None
    if legacy_row.get("is_pair_task") and legacy_row.get("text_pair"):
        pair_id = legacy_row.get("id")
        contrast_type = task

    meta = empty_metadata(
        source=legacy_row.get("source_dataset", legacy_row.get("source_name", "")),
        domain=track,
        language="ru",
        split=legacy_row.get("split", "train"),
        pair_id=pair_id,
        contrast_type=contrast_type,
        text_length=len(legacy_row.get("text", "")),
    )
    meta["task_name"] = task
    meta["track"] = track
    meta["record_id"] = legacy_row.get("id")

    return make_example(
        text=legacy_row["text"],
        y_style=y_style,
        y_semantic=y_semantic,
        metadata=meta,
        legacy=legacy_row,
    )


def probing_labels_for_track(rows: list[dict[str, Any]], track: str) -> tuple[str, list[Any]]:
    """Метки и task_type для probing по треку."""
    if not rows:
        return "classification", []
    legacy = rows[0]
    task_type = legacy.get("task_type", "classification")
    if track == "style":
        if legacy.get("label_type") == "multi_label":
            return task_type, [r.get("labels") or r.get("y_style") or [] for r in rows]
        return task_type, [r.get("label", r.get("y_style")) for r in rows]
    if track == "semantic" and task_type == "regression":
        return task_type, [r.get("label", r.get("y_semantic")) for r in rows]
    return task_type, [r.get("label", r.get("y_semantic")) for r in rows]


def normalize_text_key(text: str) -> str:
    """Ключ для дедупликации и проверки утечек."""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t
