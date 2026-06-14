"""Train/val/test splits для probing"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_pipeline import load_processed_records, save_processed_records
from src.data_schema import probing_labels_for_track
from src.leakage_protocol import enforce_leakage_protocol, group_indices_by_leakage_key
from src.paths import ProjectPaths
from src.split_diagnostics import (
    assign_groups_iterative,
    class_distribution_report,
    group_label_vector,
    log_split_diagnostics,
    validate_split_coverage,
)

logger = logging.getLogger(__name__)

Partition = Literal["train", "val", "test"]


def _labels_for_splitting(rows: list[dict[str, Any]], track: str) -> np.ndarray | None:
    if not rows:
        return None
    legacy = rows[0]
    if legacy.get("task_type") != "classification":
        return None
    if legacy.get("label_type") == "multi_label":
        return None
    _, labels = probing_labels_for_track(rows, track)
    try:
        return np.array([int(x) for x in labels], dtype=np.int64)
    except (TypeError, ValueError):
        return None


def _indices_from_group_keys(
    groups: dict[str, list[int]],
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
) -> dict[str, list[int]]:
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for key in train_keys:
        train_idx.extend(groups[key])
    for key in val_keys:
        val_idx.extend(groups[key])
    for key in test_keys:
        test_idx.extend(groups[key])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _split_groups_stratified(
    rows: list[dict[str, Any]],
    groups: dict[str, list[int]],
    track: str,
    *,
    test_size: float,
    val_size: float,
    seed: int,
    min_eval_frequency: int = 5,
) -> dict[str, list[int]]:
    """Strict stratified split по leakage-группам (single-label и multilabel)."""
    group_keys = list(groups.keys())
    group_vectors = [group_label_vector(rows, groups[k], track) for k in group_keys]
    train_keys, val_keys, test_keys = assign_groups_iterative(
        group_keys,
        group_vectors,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        min_eval_frequency=min_eval_frequency,
    )
    return _indices_from_group_keys(groups, train_keys, val_keys, test_keys)


def build_task_split(
    rows: list[dict[str, Any]],
    *,
    track: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    group_by_leakage: bool = True,
    stratify: bool = True,
    min_eval_frequency: int = 5,
) -> dict[str, list[int]]:
    """
    Индексы train/val/test.

    При group_by_leakage=True все строки одной leakage-компоненты
    попадают в один partition; stratify=True балансирует классы по группам.
    """
    if group_by_leakage:
        groups = group_indices_by_leakage_key(rows)
        group_keys = list(groups.keys())
        if stratify:
            indices = _split_groups_stratified(
                rows,
                groups,
                track,
                test_size=test_size,
                val_size=val_size,
                seed=seed,
                min_eval_frequency=min_eval_frequency,
            )
        else:
            train_keys, test_keys = train_test_split(
                group_keys,
                test_size=test_size,
                random_state=seed,
            )
            rel_val = val_size / (1.0 - test_size) if test_size < 1.0 else val_size
            train_keys, val_keys = train_test_split(
                train_keys,
                test_size=rel_val,
                random_state=seed,
            )
            indices = _indices_from_group_keys(groups, train_keys, val_keys, test_keys)
        logger.info(
            "split by leakage groups: %s groups → train=%s val=%s test=%s rows (stratify=%s)",
            len(group_keys),
            len(indices["train"]),
            len(indices["val"]),
            len(indices["test"]),
            stratify,
        )
        return indices

    indices = np.arange(len(rows))
    stratify = _labels_for_splitting(rows, track)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    stratify_train = stratify[train_idx] if stratify is not None else None
    rel_val = val_size / (1.0 - test_size) if test_size < 1.0 else val_size
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=rel_val,
        random_state=seed,
        stratify=stratify_train,
    )
    return {
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }


def _export_partition_files(
    task_rows: list[dict[str, Any]],
    indices: dict[str, list[int]],
    task_dir: Path,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for part in ("train", "val", "test"):
        subset = [task_rows[i] for i in indices[part]]
        dest = task_dir / f"{part}.jsonl"
        save_processed_records(subset, dest)
        paths[part] = str(dest)
    return paths


def build_splits_for_track(
    track: str,
    processed_path: Path,
    *,
    task_names: list[str],
    test_size: float,
    val_size: float,
    seed: int,
    output_dir: Path,
    leakage_cfg: dict[str, Any] | None = None,
    splits_cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    rows = load_processed_records(processed_path)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task = row.get("task_name") or (row.get("metadata") or {}).get("task_name")
        if task in task_names:
            by_task.setdefault(task, []).append(row)

    splits_cfg = splits_cfg or {}
    stratify = bool(splits_cfg.get("stratify", True))
    min_per_class = splits_cfg.get("min_samples_per_class") or {
        "train": 2,
        "val": 1,
        "test": 2,
    }
    fail_on_coverage = bool(splits_cfg.get("fail_on_coverage_error", True))

    saved: dict[str, Path] = {}
    for task_name, task_rows in by_task.items():
        group_by = True
        if leakage_cfg is not None:
            group_by = bool(leakage_cfg.get("group_splits", True))

        indices = build_task_split(
            task_rows,
            track=track,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            group_by_leakage=group_by,
            stratify=stratify,
            min_eval_frequency=int(
                splits_cfg.get("min_global_label_frequency_for_eval", 5)
            ),
        )

        dist = class_distribution_report(task_rows, indices, track)
        validation = validate_split_coverage(
            task_rows,
            indices,
            track,
            min_samples_per_class=min_per_class,
            min_positive_per_label=int(splits_cfg.get("min_positive_per_label", 1)),
            min_negative_per_label=int(splits_cfg.get("min_negative_per_label", 1)),
            min_global_label_frequency_for_eval=int(
                splits_cfg.get("min_global_label_frequency_for_eval", 5)
            ),
        )
        log_split_diagnostics(track, task_name, dist, validation)
        if not validation.ok and fail_on_coverage:
            raise RuntimeError(
                f"Split coverage failed for {track}/{task_name}: "
                + "; ".join(validation.errors)
            )
        if leakage_cfg:
            report = enforce_leakage_protocol(task_rows, indices, leakage_cfg)
            if report.ok:
                logger.info("Leakage %s/%s: ok", track, task_name)
            else:
                for v in report.violations:
                    logger.error("Leakage %s/%s: %s", track, task_name, v)
                if leakage_cfg.get("fail_on_violation", True):
                    raise RuntimeError(
                        "Leakage protocol failed: " + "; ".join(report.violations)
                    )
                logger.warning(
                    "Leakage %s/%s: violations present but fail_on_violation=false; splits saved",
                    track,
                    task_name,
                )

        task_dir = output_dir / track / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        partition_paths = _export_partition_files(task_rows, indices, task_dir)

        dest = task_dir / "split_manifest.json"
        payload = {
            "track": track,
            "task_name": task_name,
            "n_total": len(task_rows),
            "seed": seed,
            "test_size": test_size,
            "val_size": val_size,
            "indices": indices,
            "record_ids": [
                task_rows[i].get("id") or (task_rows[i].get("metadata") or {}).get("record_id")
                for i in range(len(task_rows))
            ],
            "partitions": partition_paths,
            "schema": "text,y_style,y_semantic,metadata",
            "group_by_leakage": group_by,
            "stratify": stratify,
            "class_distribution": dist,
            "coverage_validation": validation.to_dict(),
        }
        dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        diag_path = task_dir / "split_diagnostics.json"
        diag_path.write_text(
            json.dumps(
                {
                    "track": track,
                    "task_name": task_name,
                    "class_distribution": dist,
                    "coverage_validation": validation.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        saved[task_name] = dest
        logger.info(
            "Split %s/%s: train=%s val=%s test=%s → %s",
            track,
            task_name,
            len(indices["train"]),
            len(indices["val"]),
            len(indices["test"]),
            task_dir.resolve(),
        )
    return saved


def build_all_splits(cfg: dict[str, Any], *, paths: ProjectPaths | None = None) -> dict[str, dict[str, Path]]:
    project = paths or ProjectPaths.from_root()
    splits_cfg = cfg.get("splits") or {}
    test_size = float(splits_cfg.get("test_size", 0.2))
    val_size = float(splits_cfg.get("val_size", 0.1))
    seed = int(cfg.get("seed", 42))
    leakage_cfg = (cfg.get("data") or {}).get("leakage")
    splits_root = Path(cfg.get("artifacts", {}).get("data_splits", project.data_splits))
    processed_root = Path(cfg.get("artifacts", {}).get("data_processed", project.data_processed))

    result: dict[str, dict[str, Path]] = {}
    tracks_cfg = cfg.get("tracks") or {}
    for track, track_cfg in tracks_cfg.items():
        if not track_cfg.get("enabled", False):
            continue
        processed = processed_root / f"{track}_records.jsonl"
        if not processed.exists():
            raise FileNotFoundError(f"Processed data missing: {processed}. Run prepare_data first.")
        result[track] = build_splits_for_track(
            track,
            processed,
            task_names=list(track_cfg.get("tasks", [])),
            test_size=test_size,
            val_size=val_size,
            seed=seed,
            output_dir=splits_root,
            leakage_cfg=leakage_cfg,
            splits_cfg=splits_cfg,
        )
    return result


def split_manifest_path(
    cfg: dict[str, Any],
    *,
    track: str,
    task_name: str,
    paths: ProjectPaths | None = None,
) -> Path:
    project = paths or ProjectPaths.from_root()
    root = Path(cfg.get("artifacts", {}).get("data_splits", project.data_splits))
    return root / track / task_name / "split_manifest.json"


def load_partition(
    cfg: dict[str, Any],
    *,
    track: str,
    task_name: str,
    partition: Partition,
    paths: ProjectPaths | None = None,
) -> list[dict[str, Any]]:
    """Загрузка фиксированного split-артефакта (train/val/test.jsonl)."""
    project = paths or ProjectPaths.from_root()
    root = Path(cfg.get("artifacts", {}).get("data_splits", project.data_splits))
    part_path = root / track / task_name / f"{partition}.jsonl"
    if not part_path.exists():
        raise FileNotFoundError(f"Split artifact missing: {part_path}. Run build_splits.")
    return load_processed_records(part_path)
