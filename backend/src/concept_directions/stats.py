"""Статистики concept directions"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def direction_statistics(
    directions: np.ndarray,
    x: np.ndarray,
    *,
    other_layer_directions: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """norm, cosine similarity, projection stats, inter-layer similarity."""
    if directions.ndim == 1:
        directions = directions.reshape(1, -1)

    norms = np.linalg.norm(directions, axis=1)
    stats: dict[str, Any] = {
        "norms": norms.tolist(),
        "mean_norm": float(norms.mean()),
    }

    if directions.shape[0] > 1:
        sims = []
        for i in range(directions.shape[0]):
            for j in range(i + 1, directions.shape[0]):
                sims.append(_cosine(directions[i], directions[j]))
        stats["cosine_similarity_pairwise"] = sims
        stats["mean_cosine_similarity"] = float(np.mean(sims)) if sims else 0.0

    projections = x @ directions.T
    stats["projection_mean"] = projections.mean(axis=0).tolist()
    stats["projection_std"] = projections.std(axis=0).tolist()

    if other_layer_directions:
        cross: dict[str, float] = {}
        for layer_idx, other_dirs in other_layer_directions.items():
            if other_dirs.shape[0] == 0:
                continue
            s = _cosine(directions[0], other_dirs[0])
            cross[str(layer_idx)] = s
        stats["inter_layer_cosine"] = cross
        stats["cross_domain_transferability_proxy"] = float(np.mean(list(cross.values()))) if cross else None

    return stats
