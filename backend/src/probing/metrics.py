"""Метрики probing"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _class_distribution(y_true: np.ndarray) -> dict[str, Any]:
    if y_true.ndim == 2:
        freq = y_true.sum(axis=0).astype(int)
        return {
            "n_samples": int(y_true.shape[0]),
            "n_labels": int(y_true.shape[1]),
            "label_frequency": freq.tolist(),
            "labels_with_positive": int(sum(1 for f in freq if f > 0)),
        }
    values, counts = np.unique(y_true, return_counts=True)
    return {
        "n_samples": int(len(y_true)),
        "n_classes": int(len(values)),
        "class_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
    }


def _random_baselines(y_true: np.ndarray, task_type: str) -> dict[str, float]:
    if task_type == "regression":
        return {"mean_baseline_neg_mse": 0.0}
    if y_true.ndim == 2:
        n_labels = y_true.shape[1]
        random_f1 = 1.0 / max(n_labels, 1)
        return {"random_f1_macro": float(random_f1), "majority_f1_macro": 0.0}
    values, counts = np.unique(y_true, return_counts=True)
    majority_acc = float(counts.max() / counts.sum()) if len(counts) else 0.0
    random_acc = 1.0 / max(len(values), 1)
    return {"majority_accuracy": majority_acc, "random_accuracy": float(random_acc)}


def _safe_roc_auc_binary(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _safe_roc_auc_multilabel(y_true: np.ndarray, proba: np.ndarray) -> tuple[float | None, list[int]]:
    skipped: list[int] = []
    scores: list[float] = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if len(np.unique(col)) < 2:
            skipped.append(j)
            continue
        scores.append(float(roc_auc_score(col, proba[:, j])))
    if not scores:
        return None, skipped
    return float(np.mean(scores)), skipped


def _safe_roc_auc_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, proba, multi_class="ovr"))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    proba: np.ndarray | None = None,
    task_type: str = "classification",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"class_distribution": _class_distribution(y_true)}
    metrics["random_baselines"] = _random_baselines(y_true, task_type)

    if task_type == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        metrics["neg_mse"] = -mse
        metrics["mse"] = mse
        return metrics

    if y_true.ndim == 2:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        if proba is not None and proba.ndim == 2:
            auc, skipped = _safe_roc_auc_multilabel(y_true, proba)
            if auc is not None:
                metrics["roc_auc_macro"] = auc
            if skipped:
                metrics["roc_auc_skipped_labels"] = skipped
        metrics["confusion_matrix"] = confusion_matrix(
            y_true.argmax(axis=1), y_pred.argmax(axis=1)
        ).tolist()
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    if proba is not None:
        if proba.ndim == 2 and proba.shape[1] == 2:
            auc = _safe_roc_auc_binary(y_true, proba[:, 1])
        else:
            auc = _safe_roc_auc_multiclass(y_true, proba)
        if auc is not None:
            metrics["roc_auc"] = auc
    return metrics


def primary_score(metrics: dict[str, Any], task_type: str) -> float:
    if task_type == "regression":
        return float(metrics.get("neg_mse", -metrics.get("mse", 0.0)))
    if "f1_macro" in metrics:
        return float(metrics["f1_macro"])
    if "f1" in metrics:
        return float(metrics["f1"])
    return float(metrics.get("accuracy", 0.0))
