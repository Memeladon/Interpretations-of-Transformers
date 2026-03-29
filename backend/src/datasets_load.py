from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Literal

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

TrackName = Literal["semantic", "tone", "style"]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, list):
        return [int(v) for v in values]
    return [int(values)]


# Порядок = индекс класса для seara/ru_go_emotions (сырые бинарные столбцы без поля ``labels``).
GO_EMOTIONS_BINARY_COLUMNS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


def _binary_column_on(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return False


def _go_emotions_labels_from_row(item: dict[str, Any]) -> list[int]:
    """
    HF может отдавать список ``labels``; в ``raw`` часто только столбцы admiration, anger, …
    """
    raw_labels = item.get("labels")
    if raw_labels is not None and isinstance(raw_labels, list) and len(raw_labels) > 0:
        return sorted(set(_to_int_list(raw_labels)))

    active: list[int] = []
    for idx, col in enumerate(GO_EMOTIONS_BINARY_COLUMNS):
        if _binary_column_on(item.get(col)):
            active.append(idx)
    return sorted(set(active))


def _emotion_text_from_row(item: dict[str, Any]) -> str:
    return _safe_text(item.get("text")) or _safe_text(item.get("ru_text"))


def _sample_rows(rows: list[dict[str, Any]], limit: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if limit is None or len(rows) <= limit:
        return rows
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [rows[i] for i in selected]


def _with_common_meta(
    *,
    source_dataset: str,
    source_name: str,
    split: str,
    row_id: int,
    task_name: str,
    track: TrackName,
    task_type: str,
    label_type: str,
    text: str,
    text_pair: str | None,
    label: int | float | None,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{source_name}_{row_id}",
        "source_id": row_id,
        "source_name": source_name,
        "source_dataset": source_dataset,
        "split": split,
        "task_name": task_name,
        "track": track,
        # Обратная совместимость: старый ключ (группа смысла «семантика / не семантика»)
        "task_group": "semantic" if track == "semantic" else "supervised",
        "task_type": task_type,
        "label_type": label_type,
        "text": text,
        "text_pair": text_pair,
        "is_pair_task": text_pair is not None,
        "label": label,
        "labels": labels,
    }


def _ru_sts_mapper(
    item: dict[str, Any],
    row_id: int,
    source_name: str,
    source_dataset: str,
    split: str,
) -> dict[str, Any]:
    text = _safe_text(item.get("sentence1"))
    text_pair = _safe_text(item.get("sentence2"))
    score = float(item.get("score", 0.0))
    return {
        **_with_common_meta(
            source_dataset=source_dataset,
            source_name=source_name,
            split=split,
            row_id=row_id,
            task_name="semantic_similarity",
            track="semantic",
            task_type="regression",
            label_type="regression",
            text=text,
            text_pair=text_pair,
            label=score,
            labels=None,
        )
    }


def _ru_qqp_mapper(
    item: dict[str, Any],
    row_id: int,
    source_name: str,
    source_dataset: str,
    split: str,
) -> dict[str, Any]:
    text = _safe_text(item.get("text1"))
    text_pair = _safe_text(item.get("text2"))
    label = int(item.get("label", 0))
    return {
        **_with_common_meta(
            source_dataset=source_dataset,
            source_name=source_name,
            split=split,
            row_id=row_id,
            task_name="paraphrase",
            track="semantic",
            task_type="classification",
            label_type="single_label",
            text=text,
            text_pair=text_pair,
            label=label,
            labels=[label],
        )
    }


def _sentiment_mapper(
    item: dict[str, Any],
    row_id: int,
    source_name: str,
    source_dataset: str,
    split: str,
) -> dict[str, Any]:
    text = _safe_text(item.get("text"))
    label = int(item.get("sentiment", 0))
    return {
        **_with_common_meta(
            source_dataset=source_dataset,
            source_name=source_name,
            split=split,
            row_id=row_id,
            task_name="sentiment",
            track="tone",
            task_type="classification",
            label_type="single_label",
            text=text,
            text_pair=None,
            label=label,
            labels=[label],
        )
    }


def _emotion_mapper(
    item: dict[str, Any],
    row_id: int,
    source_name: str,
    source_dataset: str,
    split: str,
) -> dict[str, Any]:
    text = _emotion_text_from_row(item)
    labels = _go_emotions_labels_from_row(item)
    primary_label = labels[0] if labels else 0
    return {
        **_with_common_meta(
            source_dataset=source_dataset,
            source_name=source_name,
            split=split,
            row_id=row_id,
            task_name="emotion",
            track="style",
            task_type="classification",
            label_type="multi_label",
            text=text,
            text_pair=None,
            label=primary_label,
            labels=labels,
        )
    }


SEMANTIC_SPECS: list[dict[str, Any]] = [
    {
        "dataset_name": "ai-forever/ru-stsbenchmark-sts",
        "config_name": None,
        "split": "train",
        "source_name": "ru_sts",
        "mapper": _ru_sts_mapper,
    },
    {
        "dataset_name": "MilyaShams/qqp-ru_10k",
        "config_name": None,
        "split": "train",
        "source_name": "ru_qqp",
        "mapper": _ru_qqp_mapper,
    },
]

TONE_SPECS: list[dict[str, Any]] = [
    {
        "dataset_name": "MonoHime/ru_sentiment_dataset",
        "config_name": None,
        "split": "train",
        "source_name": "sentiment",
        "mapper": _sentiment_mapper,
    },
]

STYLE_SPECS: list[dict[str, Any]] = [
    {
        "dataset_name": "seara/ru_go_emotions",
        "config_name": "raw",
        "split": "train",
        "source_name": "emotion",
        "mapper": _emotion_mapper,
    },
]

DATASET_SPECS: list[dict[str, Any]] = SEMANTIC_SPECS + TONE_SPECS + STYLE_SPECS


def _load_single_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    cache_dir: str | Path,
) -> Dataset:
    return load_dataset(dataset_name, name=config_name, split=split, cache_dir=str(cache_dir))


def _normalize_dataset(
    ds: Dataset,
    source_dataset: str,
    source_name: str,
    split: str,
    mapper: Callable[[dict[str, Any], int, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        row = mapper(item, i, source_name, source_dataset, split)
        if row["text"]:
            rows.append(row)
    return rows


def _load_specs(
    specs: list[dict[str, Any]],
    *,
    cache_dir: str | Path,
    limit_per_dataset: int | None,
    seed: int,
    track_label: str,
) -> list[dict[str, Any]]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for i, spec in enumerate(specs, start=1):
        ds_label = spec["dataset_name"]
        if spec.get("config_name"):
            ds_label = f'{spec["dataset_name"]} ({spec["config_name"]})'
        logger.info(
            "[%s] dataset %s/%s: loading %s split=%s …",
            track_label,
            i,
            len(specs),
            ds_label,
            spec["split"],
        )
        ds = _load_single_dataset(
            dataset_name=spec["dataset_name"],
            config_name=spec["config_name"],
            split=spec["split"],
            cache_dir=cache_path,
        )
        normalized = _normalize_dataset(
            ds=ds,
            source_dataset=spec["dataset_name"],
            source_name=spec["source_name"],
            split=spec["split"],
            mapper=spec["mapper"],
        )
        normalized = _sample_rows(normalized, limit_per_dataset, rng)
        rows.extend(normalized)
        logger.info(
            "[%s] dataset %s/%s: done rows=%s (total=%s)",
            track_label,
            i,
            len(specs),
            len(normalized),
            len(rows),
        )

    rng.shuffle(rows)
    return rows


def load_track_datasets(
    track: TrackName,
    cache_dir: str | Path = ".cache/hf_datasets",
    limit_per_dataset: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if track == "semantic":
        specs = SEMANTIC_SPECS
    elif track == "tone":
        specs = TONE_SPECS
    elif track == "style":
        specs = STYLE_SPECS
    else:
        raise ValueError(f"Unknown track: {track}")
    return _load_specs(specs, cache_dir=cache_dir, limit_per_dataset=limit_per_dataset, seed=seed, track_label=track)


def load_all_datasets(
    cache_dir: str | Path = ".cache/hf_datasets",
    limit_per_dataset: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Все треки подряд (для отладки). Порядок: semantic, tone, style."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for spec_list, label in ((SEMANTIC_SPECS, "semantic"), (TONE_SPECS, "tone"), (STYLE_SPECS, "style")):
        part = _load_specs(
            spec_list,
            cache_dir=cache_path,
            limit_per_dataset=limit_per_dataset,
            seed=seed,
            track_label=label,
        )
        rows.extend(part)

    rng.shuffle(rows)
    return rows
