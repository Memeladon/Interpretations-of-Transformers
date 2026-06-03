"""Анализ устойчивости layer-wise probing"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.probing import train_probes_by_layer

logger = logging.getLogger(__name__)


def _load_layer_matrices(embedding_dir: Path, prefix: str) -> list[np.ndarray]:
    paths = sorted(
        embedding_dir.glob(f"{prefix}_layer_*.npy"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    matrices: list[np.ndarray] = []
    for path in paths:
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            rows = []
            for sample in arr:
                sample_np = np.asarray(sample)
                rows.append(sample_np.mean(axis=0) if sample_np.ndim > 1 else sample_np)
            matrices.append(np.stack(rows, axis=0))
        else:
            matrices.append(np.asarray(arr))
    return matrices


def analyze_probe_robustness(
    *,
    layer_matrices: list[np.ndarray],
    labels: list[Any],
    task_type: str,
    bootstrap_seeds: list[int],
    subsample_fraction: float = 0.8,
    probing_n_jobs: int = 1,
) -> dict[str, Any]:
    n = len(labels)
    subsample_n = max(2, int(n * subsample_fraction))
    per_seed_scores: dict[str, dict[str, float]] = {}
    for seed in bootstrap_seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=subsample_n, replace=False)
        sub_labels = [labels[i] for i in idx]
        sub_layers = [mat[idx] for mat in layer_matrices]
        scores = train_probes_by_layer(
            sub_layers,
            sub_labels,
            task_type=task_type,
            seed=seed,
            n_jobs=probing_n_jobs,
        )
        per_seed_scores[str(seed)] = {str(k): float(v) for k, v in scores.items()}

    layer_keys = sorted(
        {k for scores in per_seed_scores.values() for k in scores},
        key=lambda x: int(x) if x.isdigit() else 999,
    )
    summary: dict[str, Any] = {"per_seed": per_seed_scores, "layers": {}}
    for layer in layer_keys:
        vals = [per_seed_scores[s][layer] for s in per_seed_scores if layer in per_seed_scores[s]]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        summary["layers"][layer] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": len(vals),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return summary


def robustness_from_analysis_summaries(
    analysis_root: Path,
    *,
    bootstrap_seeds: list[int],
    subsample_fraction: float,
    probing_n_jobs: int = 1,
) -> dict[str, Any]:
    """Обход pipeline_summary.json и повторный probing на сохранённых эмбеддингах."""
    results: dict[str, Any] = {}
    for summary_path in analysis_root.rglob("pipeline_summary.json"):
        rel = summary_path.parent.relative_to(analysis_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        track, task, family, level = parts[0], parts[1], parts[2], parts[3]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        embedding_paths = summary.get("embedding_paths") or []
        if not embedding_paths:
            continue
        prefix = Path(embedding_paths[0]).stem.rsplit("_layer_", 1)[0]
        layer_matrices = _load_layer_matrices(summary_path.parent, prefix)
        meta_path = summary_path.parent / "task_meta.json"
        if not meta_path.exists():
            logger.warning("Skip robustness (no task_meta.json): %s", summary_path.parent)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        key = f"{track}/{task}/{family}/{level}"
        results[key] = analyze_probe_robustness(
            layer_matrices=layer_matrices,
            labels=meta["labels"],
            task_type=meta["task_type"],
            bootstrap_seeds=bootstrap_seeds,
            subsample_fraction=subsample_fraction,
            probing_n_jobs=probing_n_jobs,
        )
    return results
