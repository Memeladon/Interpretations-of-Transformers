from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def classification_y_array(labels: list[Any]) -> np.ndarray:
    """Single-label: shape (N,). Multi-label from списков индексов: shape (N, C) float32 multi-hot."""
    if not labels:
        return np.array([], dtype=np.float64)
    if isinstance(labels[0], (list, tuple)):
        rows = [[int(x) for x in row] for row in labels]
        flat = [x for row in rows for x in row]
        c = max(flat) + 1 if flat else 1
        y = np.zeros((len(rows), c), dtype=np.float32)
        for i, row in enumerate(rows):
            for j in row:
                if 0 <= j < c:
                    y[i, j] = 1.0
        return y
    return np.array([int(x) for x in labels], dtype=np.int64)


def _build_probe(task_type: str, seed: int = 42, *, n_jobs: int = 1):
    if task_type == "classification":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    if task_type == "regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", Ridge(random_state=seed)),
            ]
        )
    raise ValueError(f"Unsupported task type: {task_type}")


def _score(task_type: str, y_true, pred) -> float:
    if task_type == "classification":
        return float(accuracy_score(y_true, pred))
    return float(-mean_squared_error(y_true, pred))


def train_probes_by_layer(
    layer_outputs: list[Any],
    labels: list[int | float] | list[list[int]],
    task_type: str,
    test_size: float = 0.2,
    seed: int = 42,
    *,
    n_jobs: int = 1,
) -> dict[str, Any]:
    if task_type == "classification":
        y = classification_y_array(list(labels))
    else:
        y = np.array(labels, dtype=np.float64)

    results: dict[str, Any] = {}
    stratify = None
    if task_type == "classification" and y.ndim == 1:
        stratify = y

    for layer_idx, x_layer in enumerate(layer_outputs):
        x = x_layer if isinstance(x_layer, np.ndarray) else x_layer.numpy()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=stratify
        )

        if task_type == "classification" and y.ndim == 2:
            model = OneVsRestClassifier(
                _build_probe(task_type="classification", seed=seed, n_jobs=n_jobs)
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = float(f1_score(y_test, pred, average="macro", zero_division=0))
        else:
            model = _build_probe(task_type=task_type, seed=seed, n_jobs=n_jobs)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = _score(task_type, y_test, pred)

        results[str(layer_idx)] = score
        logger.debug("probe layer %s: score=%.4f", layer_idx, score)
    return results
