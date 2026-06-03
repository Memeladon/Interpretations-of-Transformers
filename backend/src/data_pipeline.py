"""Подготовка данных"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from src.data_loaders import load_from_source_spec, load_track
from src.data_preprocessing import preprocess_records
from src.paths import ProjectPaths

logger = logging.getLogger(__name__)

TrackList = list[Literal["semantic", "tone", "style"]]


def _enabled_tracks(cfg: dict[str, Any]) -> TrackList:
    tracks_cfg = cfg.get("tracks") or {}
    out: TrackList = []
    for name in ("semantic", "tone", "style"):
        if tracks_cfg.get(name, {}).get("enabled", False):
            out.append(name)  # type: ignore[arg-type]
    return out


def _preprocessing_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return (cfg.get("data") or {}).get("preprocessing") or {}


def _track_source_spec(cfg: dict[str, Any], track: str) -> dict[str, Any]:
    sources = (cfg.get("data") or {}).get("sources") or {}
    per_track = sources.get("tracks") or {}
    if track in per_track:
        return per_track[track]
    default = sources.get("default", "huggingface")
    if isinstance(default, str):
        return {"type": default}
    return default if isinstance(default, dict) else {"type": "huggingface"}


def prepare_track_records(
    track: Literal["semantic", "tone", "style"],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = _track_source_spec(cfg, track)
    cache_dir = cfg.get("cache_dir", ".cache/hf_datasets")
    limit = cfg.get("dataset_limit_per_source")
    seed = int(cfg.get("seed", 42))
    records = load_from_source_spec(
        spec,
        cache_dir=cache_dir,
        limit_per_dataset=limit,
        seed=seed,
        track=track,
    )
    prep = _preprocessing_cfg(cfg)
    return preprocess_records(
        records,
        version=str(prep.get("version", "1.0")),
        clean=bool(prep.get("clean_text", True)),
        normalize_ws=bool(prep.get("normalize_whitespace", True)),
        deduplicate=bool(prep.get("deduplicate", True)),
        filter_empty=bool(prep.get("filter_empty", True)),
        max_text_length=prep.get("max_text_length"),
        balance_classes=bool(prep.get("balance_classes", False)),
        balance_max_per_class=prep.get("balance_max_per_class"),
        seed=seed,
    )


def save_processed_records(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Saved %s records → %s", len(records), path.resolve())
    return path


def load_processed_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _processed_path(processed_root: Path, track: str) -> Path:
    return processed_root / f"{track}_records.jsonl"


def _latest_raw_dump_dir(paths: ProjectPaths) -> Path | None:
    dump_root = paths.data_raw / "dataset_dump"
    if not dump_root.is_dir():
        return None
    candidates = [
        p for p in dump_root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _load_track_from_raw_dump(
    track: Literal["semantic", "tone", "style"],
    dump_dir: Path,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    src = dump_dir / f"{track}_records.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"No dump for track {track}: {src}")
    prep = _preprocessing_cfg(cfg)
    seed = int(cfg.get("seed", 42))
    records = load_processed_records(src)
    return preprocess_records(
        records,
        version=str(prep.get("version", "1.0")),
        clean=bool(prep.get("clean_text", True)),
        normalize_ws=bool(prep.get("normalize_whitespace", True)),
        deduplicate=bool(prep.get("deduplicate", True)),
        filter_empty=bool(prep.get("filter_empty", True)),
        max_text_length=prep.get("max_text_length"),
        balance_classes=bool(prep.get("balance_classes", False)),
        balance_max_per_class=prep.get("balance_max_per_class"),
        seed=seed,
    )


def ensure_processed_tracks(
    cfg: dict[str, Any],
    *,
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Подготовить data/processed/*, пропуская уже существующие файлы."""
    project = paths or ProjectPaths.from_root()
    project.ensure_dirs()
    processed_root = Path(cfg.get("artifacts", {}).get("data_processed", project.data_processed))
    dump_dir = _latest_raw_dump_dir(project)
    out: dict[str, Path] = {}

    for track in _enabled_tracks(cfg):
        dest = _processed_path(processed_root, track)
        if dest.exists() and not force:
            logger.info("skip prepare: %s already exists", dest.resolve())
            out[track] = dest
            continue

        try:
            records = prepare_track_records(track, cfg)
        except Exception as exc:
            if dump_dir is None:
                raise
            logger.warning(
                "HF load failed for track=%s (%s); falling back to raw dump %s",
                track,
                exc,
                dump_dir.resolve(),
            )
            records = _load_track_from_raw_dump(track, dump_dir, cfg)

        save_processed_records(records, dest)
        out[track] = dest
    return out


def prepare_all_tracks(
    cfg: dict[str, Any],
    *,
    paths: ProjectPaths | None = None,
    force: bool = False,
) -> dict[str, Path]:
    return ensure_processed_tracks(cfg, paths=paths, force=force)
