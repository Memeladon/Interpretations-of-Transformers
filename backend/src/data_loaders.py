"""Загрузка датасетов: CSV, JSON, JSONL, TXT, HuggingFace (по умолчанию)"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal

from src.data_schema import legacy_from_track_mapper, make_example
from src.datasets_load import TrackName, load_track_datasets

logger = logging.getLogger(__name__)

SourceKind = Literal["huggingface", "csv", "json", "jsonl", "txt"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return list(data["records"])
    raise ValueError(f"JSON must be a list or {{records: [...]}}, got {type(data)}")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_txt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text:
                rows.append({"text": text, "id": f"line_{i}"})
    return rows


def _row_to_tz(raw: dict[str, Any], *, default_source: str) -> dict[str, Any]:
    """Строка файла → схема ТЗ (если уже TZ — вернуть как есть)."""
    if "y_style" in raw or "y_semantic" in raw:
        meta = raw.get("metadata") or {}
        meta.setdefault("source", default_source)
        return make_example(
            text=str(raw.get("text", "")),
            y_style=raw.get("y_style"),
            y_semantic=raw.get("y_semantic"),
            metadata=meta,
        )
    text = str(raw.get("text", raw.get("sentence", "")))
    return make_example(
        text=text,
        y_style=raw.get("y_style", raw.get("style_label")),
        y_semantic=raw.get("y_semantic", raw.get("semantic_label", raw.get("label"))),
        metadata={
            "source": raw.get("source", default_source),
            "domain": raw.get("domain", ""),
            "language": raw.get("language", "ru"),
            "split": raw.get("split", "train"),
            "pair_id": raw.get("pair_id"),
            "contrast_type": raw.get("contrast_type"),
            "preprocessing_version": raw.get("preprocessing_version", "1.0"),
            "text_length": len(text),
        },
    )


def load_file_dataset(path: str | Path, *, kind: SourceKind | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    kind = kind or {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".txt": "txt",
    }.get(suffix, "jsonl")  # type: ignore[assignment]

    if kind == "csv":
        raw_rows = _read_csv(p)
    elif kind == "json":
        raw_rows = _read_json(p)
    elif kind == "jsonl":
        raw_rows = _read_jsonl(p)
    elif kind == "txt":
        raw_rows = _read_txt(p)
    else:
        raise ValueError(f"Unsupported file kind: {kind}")

    return [_row_to_tz(r, default_source=p.name) for r in raw_rows]


def load_track(
    track: TrackName,
    *,
    source: SourceKind = "huggingface",
    path: str | Path | None = None,
    cache_dir: str | Path = ".cache/hf_datasets",
    limit_per_dataset: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if source == "huggingface":
        legacy_rows = load_track_datasets(
            track, cache_dir=cache_dir, limit_per_dataset=limit_per_dataset, seed=seed
        )
        return [legacy_from_track_mapper(r) for r in legacy_rows]
    if path is None:
        raise ValueError(f"path required for source={source}")
    file_rows = load_file_dataset(path, kind=source)
    for i, row in enumerate(file_rows):
        meta = row["metadata"]
        row["id"] = meta.get("record_id", f"{track}_{i}")
        row["task_name"] = meta.get("task_name", track)
        row["track"] = track
        row["task_type"] = "classification"
        row["label_type"] = "single_label"
        row["label"] = row.get("y_semantic")
        row["labels"] = row.get("y_style") if isinstance(row.get("y_style"), list) else None
        row["text_pair"] = None
        row["is_pair_task"] = False
    return file_rows


def load_from_source_spec(
    spec: dict[str, Any],
    *,
    cache_dir: str | Path,
    limit_per_dataset: int | None,
    seed: int,
    track: TrackName,
) -> list[dict[str, Any]]:
    kind: SourceKind = spec.get("type", "huggingface")
    return load_track(
        track,
        source=kind,
        path=spec.get("path"),
        cache_dir=cache_dir,
        limit_per_dataset=limit_per_dataset,
        seed=seed,
    )
