"""Диагностика и валидация train/val/test splits."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.data_schema import probing_labels_for_track
from src.probing.layer_probe import classification_y_array

logger = logging.getLogger(__name__)

Partition = Literal["train", "val", "test"]


@dataclass
class SplitValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    partitions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "partitions": self.partitions,
        }


def _label_matrix(rows: list[dict[str, Any]], track: str) -> tuple[str, np.ndarray, list[Any]]:
    task_type, raw_labels = probing_labels_for_track(rows, track)
    if task_type == "classification" and raw_labels and isinstance(raw_labels[0], (list, tuple)):
        y = classification_y_array(list(raw_labels))
        return task_type, y, raw_labels
    if task_type == "classification":
        y = np.array([int(x) for x in raw_labels], dtype=np.int64)
        return task_type, y, raw_labels
    y = np.asarray(raw_labels, dtype=np.float64)
    return task_type, y, raw_labels


def group_label_vector(
    rows: list[dict[str, Any]],
    member_indices: list[int],
    track: str,
) -> np.ndarray:
    """Одна метка/вектор на leakage-группу (max-pool для multilabel)."""
    sub = [rows[i] for i in member_indices]
    task_type, y, _ = _label_matrix(sub, track)
    if task_type == "regression":
        return np.array([float(np.median(y))], dtype=np.float64)
    if y.ndim == 2:
        return np.clip(y.max(axis=0), 0, 1).astype(np.float64)
    values, counts = np.unique(y, return_counts=True)
    return np.array([int(values[int(np.argmax(counts))])], dtype=np.float64)


def class_distribution_report(
    rows: list[dict[str, Any]],
    indices: dict[str, list[int]],
    track: str,
) -> dict[str, Any]:
    task_type, y_full, _ = _label_matrix(rows, track)
    report: dict[str, Any] = {"task_type": task_type, "partitions": {}}

    for part, idxs in indices.items():
        if not idxs:
            report["partitions"][part] = {"n": 0}
            continue
        y = y_full[idxs]
        part_info: dict[str, Any] = {"n": len(idxs)}

        if task_type == "regression":
            part_info["label_min"] = float(np.min(y))
            part_info["label_max"] = float(np.max(y))
            part_info["label_mean"] = float(np.mean(y))
            part_info["n_unique"] = int(len(np.unique(y)))
        elif y.ndim == 2:
            freq = y.sum(axis=0).astype(int).tolist()
            part_info["n_classes"] = y.shape[1]
            part_info["label_frequency"] = freq
            part_info["labels_with_positive"] = int(sum(1 for f in freq if f > 0))
            part_info["labels_with_positive_and_negative"] = int(
                sum(1 for j in range(y.shape[1]) if y[:, j].sum() > 0 and y[:, j].sum() < len(y))
            )
        else:
            counts = Counter(int(v) for v in y)
            part_info["n_classes"] = len(counts)
            part_info["class_counts"] = {str(k): v for k, v in sorted(counts.items())}

        report["partitions"][part] = part_info

    return report


def validate_split_coverage(
    rows: list[dict[str, Any]],
    indices: dict[str, list[int]],
    track: str,
    *,
    min_samples_per_class: dict[str, int] | None = None,
    min_positive_per_label: int = 1,
    min_negative_per_label: int = 1,
    min_global_label_frequency_for_eval: int = 5,
    require_all_classes_in_train: bool = True,
    require_eval_partition_coverage: bool = True,
) -> SplitValidationReport:
    """Проверка покрытия классов; ошибки блокируют эксперимент."""
    min_samples_per_class = min_samples_per_class or {"train": 2, "val": 1, "test": 2}
    task_type, y_full, _ = _label_matrix(rows, track)
    report = SplitValidationReport()
    dist = class_distribution_report(rows, indices, track)
    report.partitions = dist["partitions"]

    if task_type == "regression":
        for part in ("val", "test"):
            n = dist["partitions"].get(part, {}).get("n", 0)
            if n < 2:
                report.errors.append(f"{part}: too few regression samples ({n})")
        report.ok = len(report.errors) == 0
        return report

    if y_full.ndim == 2:
        n_labels = y_full.shape[1]
        global_freq = y_full.sum(axis=0)
        for part in ("val", "test") if require_eval_partition_coverage else ():
            idxs = indices.get(part, [])
            if not idxs:
                report.errors.append(f"{part}: empty partition")
                continue
            y_part = y_full[idxs]
            strict_part = part == "test"
            for j in range(n_labels):
                if global_freq[j] < min_global_label_frequency_for_eval:
                    report.warnings.append(
                        f"label {j}: only {int(global_freq[j])} positive(s) globally — "
                        f"below min_global_label_frequency_for_eval={min_global_label_frequency_for_eval}; "
                        "excluded from strict eval coverage"
                    )
                    continue
                if global_freq[j] < min_positive_per_label + min_negative_per_label:
                    report.warnings.append(
                        f"label {j}: only {int(global_freq[j])} positive(s) globally — "
                        "cannot satisfy eval coverage; exclude from ROC-AUC"
                    )
                    continue
                pos = int(y_part[:, j].sum())
                neg = len(y_part) - pos
                if pos < min_positive_per_label or neg < min_negative_per_label:
                    msg = (
                        f"{part}: label {j} has pos={pos}, neg={neg} "
                        f"(need pos>={min_positive_per_label}, neg>={min_negative_per_label})"
                    )
                    if strict_part:
                        report.errors.append(msg)
                    else:
                        report.warnings.append(msg)
        report.ok = len(report.errors) == 0
        return report

    all_classes = set(int(x) for x in y_full)
    for cls in sorted(all_classes):
        total = int((y_full == cls).sum())
        if total < sum(min_samples_per_class.get(p, 0) for p in ("train", "val", "test")):
            report.warnings.append(
                f"class {cls}: only {total} sample(s) globally — strict stratification impossible"
            )

    for part, min_n in min_samples_per_class.items():
        idxs = indices.get(part, [])
        if not idxs and part != "val":
            report.errors.append(f"{part}: empty partition")
            continue
        y_part = y_full[idxs] if idxs else np.array([], dtype=np.int64)
        counts = Counter(int(v) for v in y_part)
        for cls in all_classes:
            if require_all_classes_in_train and part == "train" and counts.get(cls, 0) == 0:
                if int((y_full == cls).sum()) > 0:
                    report.errors.append(f"train: missing class {cls}")
            if part in ("val", "test") and require_eval_partition_coverage:
                n_cls = counts.get(cls, 0)
                if n_cls < min_n and int((y_full == cls).sum()) >= min_n:
                    report.errors.append(
                        f"{part}: class {cls} has {n_cls} sample(s), need >={min_n}"
                    )
                if n_cls > 0 and n_cls < min_n:
                    pass  # already reported
                elif int((y_full == cls).sum()) >= min_n and n_cls == 0:
                    report.errors.append(f"{part}: class {cls} absent despite global count >= {min_n}")

        if part in ("val", "test") and len(counts) == 1 and len(y_part) > 0:
            report.errors.append(
                f"{part}: only one class present ({list(counts.keys())[0]}) — metrics invalid"
            )

    report.ok = len(report.errors) == 0
    return report


def assign_groups_iterative(
    group_keys: list[str],
    group_vectors: list[np.ndarray],
    *,
    test_size: float,
    val_size: float,
    seed: int,
    min_eval_frequency: int = 5,
) -> tuple[list[str], list[str], list[str]]:
    """
    Итеративное назначение групп в train/val/test с балансировкой меток.
    Работает для single-label и multilabel (бинарные векторы).
    """
    n = len(group_keys)
    if n == 0:
        return [], [], []

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    n_train = max(1, n - n_test - n_val)
    targets = {"train": n_train, "val": n_val, "test": n_test}

    max_dim = max(len(v) for v in group_vectors)
    vectors = []
    for v in group_vectors:
        if len(v) < max_dim:
            padded = np.zeros(max_dim, dtype=np.float64)
            padded[: len(v)] = v
            vectors.append(padded)
        else:
            vectors.append(v.astype(np.float64))

    label_totals = np.stack(vectors, axis=0).sum(axis=0)
    partitions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    part_counts: dict[str, np.ndarray] = {
        p: np.zeros(max_dim, dtype=np.float64) for p in partitions
    }
    part_sizes: dict[str, int] = {p: 0 for p in partitions}
    unassigned = set(range(n))

    def _assign(idx: int, part: str) -> None:
        partitions[part].append(group_keys[idx])
        part_counts[part] += vectors[idx]
        part_sizes[part] += 1
        unassigned.discard(idx)

    def _test_covers_label(label_idx: int) -> bool:
        pos = part_counts["test"][label_idx]
        neg = part_sizes["test"] - pos
        return pos >= 1 and neg >= 1

    eval_labels = [
        j
        for j in range(max_dim)
        if label_totals[j] >= min_eval_frequency and label_totals[j] < len(vectors)
    ]
    for label_idx in sorted(eval_labels, key=lambda j: label_totals[j]):
        if _test_covers_label(label_idx):
            continue
        candidates = [i for i in unassigned if vectors[i][label_idx] > 0]
        if not candidates:
            continue
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        _assign(chosen, "test")

    def _imbalance(part: str, vec: np.ndarray) -> float:
        if part_sizes[part] >= targets[part]:
            return float("inf")
        score = float(part_sizes[part]) / max(targets[part], 1)
        after = part_counts[part] + vec
        for dim in range(max_dim):
            if label_totals[dim] <= 0:
                continue
            ideal = label_totals[dim] * (targets[part] / n)
            score += abs(after[dim] - ideal) / max(ideal, 1.0)
        return score

    order = list(rng.permutation(n))
    for idx in order:
        if idx not in unassigned:
            continue
        vec = vectors[idx]
        best_part = min(partitions.keys(), key=lambda p: _imbalance(p, vec))
        _assign(idx, best_part)

    return partitions["train"], partitions["val"], partitions["test"]


def log_split_diagnostics(
    track: str,
    task_name: str,
    dist: dict[str, Any],
    validation: SplitValidationReport,
) -> None:
    logger.info("Split diagnostics %s/%s:", track, task_name)
    for part, info in dist.get("partitions", {}).items():
        if dist.get("task_type") == "classification" and "class_counts" in info:
            logger.info("  %s: n=%s classes=%s %s", part, info["n"], info["n_classes"], info["class_counts"])
        elif "label_frequency" in info:
            pos = info.get("labels_with_positive", 0)
            logger.info(
                "  %s: n=%s multilabel classes=%s with_positive=%s evaluable=%s",
                part,
                info["n"],
                info.get("n_classes"),
                pos,
                info.get("labels_with_positive_and_negative"),
            )
        else:
            logger.info("  %s: %s", part, info)
    for w in validation.warnings:
        logger.warning("Split %s/%s: %s", track, task_name, w)
    for e in validation.errors:
        logger.error("Split %s/%s: %s", track, task_name, e)
