"""Layer-wise probing: P_f(l), метрики, сохранение весов и отчётов"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from src.probing.metrics import evaluate_predictions, primary_score
from src.probing.models import build_probe_pipeline

logger = logging.getLogger(__name__)


def classification_y_array(labels: list[Any]) -> np.ndarray:
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


class LayerProbeRunner:
    def __init__(
        self,
        *,
        task_type: str,
        estimator: str = "logistic",
        seed: int = 42,
        test_size: float = 0.2,
        max_iter: int = 2000,
        scaler: str = "standard",
        n_jobs: int = 1,
        output_dir: Path | None = None,
        strict_test_coverage: bool = True,
    ):
        self.task_type = task_type
        self.estimator = estimator
        self.seed = seed
        self.test_size = test_size
        self.max_iter = max_iter
        self.scaler = scaler
        self.n_jobs = n_jobs
        self.output_dir = Path(output_dir) if output_dir else None
        self.strict_test_coverage = strict_test_coverage

    def _prepare_y(self, labels: list[Any]) -> np.ndarray:
        if self.task_type == "classification":
            return classification_y_array(list(labels))
        return np.asarray(labels, dtype=np.float64)

    def run_layer(
        self,
        x: np.ndarray,
        y: np.ndarray,
        layer_idx: int,
        *,
        train_indices: list[int] | None = None,
        test_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        use_fixed = train_indices is not None and test_indices is not None
        if use_fixed:
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
        else:
            stratify = y if self.task_type == "classification" and y.ndim == 1 else None
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=self.test_size, random_state=self.seed, stratify=stratify
            )

        if self.task_type == "classification":
            if y_test.ndim == 1 and len(np.unique(y_test)) < 2:
                msg = (
                    f"Layer {layer_idx}: test split has single class {np.unique(y_test).tolist()} "
                    "— ROC-AUC/F1 invalid"
                )
                logger.error(msg)
                if self.strict_test_coverage:
                    raise ValueError(msg)
            elif y_test.ndim == 2:
                evaluable = sum(
                    1
                    for j in range(y_test.shape[1])
                    if y_test[:, j].sum() > 0 and y_test[:, j].sum() < len(y_test)
                )
                logger.info(
                    "Layer %s: multilabel test n=%s evaluable_labels=%s/%s",
                    layer_idx,
                    len(y_test),
                    evaluable,
                    y_test.shape[1],
                )

        logger.info(
            "Layer %s: train n=%s, test n=%s",
            layer_idx,
            len(y_train),
            len(y_test),
        )

        multilabel = self.task_type == "classification" and y.ndim == 2
        if multilabel:
            base = build_probe_pipeline(
                "classification",
                estimator=self.estimator,
                seed=self.seed,
                max_iter=self.max_iter,
                scaler=self.scaler,
                n_jobs=self.n_jobs,
            )
            model = OneVsRestClassifier(base)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            proba = None
        else:
            model = build_probe_pipeline(
                self.task_type,
                estimator=self.estimator,
                seed=self.seed,
                max_iter=self.max_iter,
                scaler=self.scaler,
                n_jobs=self.n_jobs,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            proba = model.predict_proba(x_test) if self.task_type == "classification" else None

        metrics = evaluate_predictions(y_test, pred, proba=proba, task_type=self.task_type)
        score = primary_score(metrics, self.task_type)

        if self.output_dir:
            layer_dir = self.output_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, layer_dir / "probe_model.joblib")
            (layer_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {"layer": layer_idx, "metrics": metrics, "primary_score": score, "P_f_l": score}


def train_probes_by_layer(
    layer_outputs: list[Any],
    labels: list[int | float] | list[list[int]],
    task_type: str,
    test_size: float = 0.2,
    seed: int = 42,
    *,
    n_jobs: int = 1,
    max_iter: int = 2000,
    scaler: str = "standard",
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    estimator: str = "logistic",
    output_dir: Path | None = None,
    strict_test_coverage: bool = True,
) -> dict[str, Any]:
    """Обратная совместимость: словарь layer → primary score + полный отчёт при output_dir."""
    runner = LayerProbeRunner(
        task_type=task_type,
        estimator=estimator,
        seed=seed,
        test_size=test_size,
        max_iter=max_iter,
        scaler=scaler,
        n_jobs=n_jobs,
        output_dir=output_dir,
        strict_test_coverage=strict_test_coverage,
    )
    y = runner._prepare_y(list(labels))
    profile: dict[str, float] = {}
    full: dict[str, Any] = {"layers": {}, "profile_P_f_l": {}}

    for layer_idx, x_layer in enumerate(layer_outputs):
        x = x_layer if isinstance(x_layer, np.ndarray) else x_layer.numpy()
        layer_result = runner.run_layer(
            x, y, layer_idx, train_indices=train_indices, test_indices=test_indices
        )
        key = str(layer_idx)
        profile[key] = layer_result["primary_score"]
        full["layers"][key] = layer_result
        full["profile_P_f_l"][key] = layer_result["P_f_l"]

    if output_dir:
        (Path(output_dir) / "layer_profile.json").write_text(
            json.dumps(full, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (Path(output_dir) / "training_log.json").write_text(
            json.dumps({"estimator": estimator, "task_type": task_type, "seed": seed}, indent=2),
            encoding="utf-8",
        )

    return profile
