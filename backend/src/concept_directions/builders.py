"""Concept directions"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

from src.probing.layer_probe import classification_y_array


def _looks_continuous(y: np.ndarray) -> bool:
    """Эвристика: если уникальных значений много — это регрессия."""
    y1 = y.reshape(-1)
    if y1.size == 0:
        return False
    try:
        uniq = np.unique(y1)
    except Exception:
        return True
    return uniq.size > 20


def probe_weight_directions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Направления по весам линейной модели.
    - classification: LogisticRegression coef_
    - regression: Ridge coef_
    """
    if y.ndim == 2:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(x, y.argmax(axis=1))
        coef = clf.coef_
    else:
        if _looks_continuous(y):
            reg = Ridge()
            reg.fit(x, y.reshape(-1))
            coef = np.asarray(reg.coef_, dtype=np.float64).reshape(1, -1)
        else:
            clf = LogisticRegression(max_iter=2000)
            clf.fit(x, y.reshape(-1))
            coef = clf.coef_

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    norms = np.linalg.norm(coef, axis=1, keepdims=True) + 1e-12
    return coef / norms


def mean_difference_directions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.argmax(axis=1)
    classes = np.unique(y)
    dirs = []
    global_mean = x.mean(axis=0)
    for c in classes:
        class_mean = x[y == c].mean(axis=0)
        d = class_mean - global_mean
        n = np.linalg.norm(d) + 1e-12
        dirs.append(d / n)
    return np.stack(dirs, axis=0)


def pca_directions(x: np.ndarray, n_components: int = 8) -> np.ndarray:
    pca = PCA(n_components=min(n_components, x.shape[1], x.shape[0]))
    pca.fit(x)
    return pca.components_


def class_separation_vectors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.argmax(axis=1)
    classes = np.unique(y)
    if len(classes) < 2:
        return np.zeros((1, x.shape[1]))
    c0 = x[y == classes[0]].mean(axis=0)
    c1 = x[y == classes[-1]].mean(axis=0)
    d = c1 - c0
    d = d / (np.linalg.norm(d) + 1e-12)
    return d.reshape(1, -1)


def build_concept_directions(
    x: np.ndarray,
    labels: list[Any],
    *,
    methods: list[str] | None = None,
    pca_components: int = 8,
) -> dict[str, Any]:
    methods = methods or ["probe_weight", "mean_difference", "pca", "class_separation"]
    y = classification_y_array(list(labels)) if labels and isinstance(labels[0], (list, int)) else np.asarray(labels)
    if not isinstance(y, np.ndarray) or y.dtype == object:
        y = np.asarray(labels, dtype=np.float64)

    out: dict[str, Any] = {}
    if "probe_weight" in methods or "probe_directions" in methods:
        dirs = probe_weight_directions(x, y if y.ndim else y.reshape(-1))
        out["probe_weight"] = {"components": dirs.tolist(), "n_directions": int(dirs.shape[0])}
    # mean_difference / class_separation имеют смысл только для классификации
    if "mean_difference" in methods and not (y.ndim == 1 and _looks_continuous(y)):
        dirs = mean_difference_directions(x, y)
        out["mean_difference"] = {"components": dirs.tolist(), "n_directions": int(dirs.shape[0])}
    if "pca" in methods:
        dirs = pca_directions(x, pca_components)
        out["pca"] = {"components": dirs.tolist(), "n_directions": int(dirs.shape[0])}
    if "class_separation" in methods and not (y.ndim == 1 and _looks_continuous(y)):
        dirs = class_separation_vectors(x, y)
        out["class_separation"] = {"components": dirs.tolist(), "n_directions": int(dirs.shape[0])}
    return out
