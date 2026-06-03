"""Baseline-классификаторы только на поверхностных признаках"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.features.surface import SurfaceFeatureExtractor
from src.probing import classification_y_array
from src.probing.metrics import evaluate_predictions, primary_score

logger = logging.getLogger(__name__)


def _build_baseline_classifier(
    task_type: str,
    *,
    estimator: str,
    seed: int,
    max_iter: int,
) -> Pipeline:
    if task_type == "classification":
        if estimator == "logistic":
            clf = LogisticRegression(max_iter=max_iter, random_state=seed)
        elif estimator == "svm":
            clf = LinearSVC(max_iter=max_iter, random_state=seed, dual="auto")
        else:
            raise ValueError(f"Unknown classification estimator: {estimator}")
    elif task_type == "regression":
        if estimator in ("ridge", "linear"):
            clf = Ridge(random_state=seed)
        else:
            raise ValueError(f"Unknown regression estimator: {estimator}")
    else:
        raise ValueError(task_type)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def run_leakage_baselines(
    records: list[dict[str, Any]],
    labels: list[Any],
    task_type: str,
    *,
    output_dir: Path,
    estimators: list[str] | None = None,
    domain_vocabulary: list[str] | None = None,
    seed: int = 42,
    test_size: float = 0.2,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
) -> dict[str, Any]:
    estimators = estimators or ["logistic"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = SurfaceFeatureExtractor(domain_vocabulary=domain_vocabulary)
    x = extractor.extract_batch(records)
    if task_type == "classification":
        y = classification_y_array(list(labels))
    else:
        y = np.asarray(labels, dtype=np.float64)

    use_fixed = train_indices is not None and test_indices is not None
    results: dict[str, Any] = {"estimators": {}, "feature_dim": int(x.shape[1])}

    for est in estimators:
        est_key = est if task_type == "classification" else "ridge"
        if use_fixed:
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
        else:
            stratify = y if task_type == "classification" and y.ndim == 1 else None
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=seed, stratify=stratify
            )

        if task_type == "classification" and y.ndim == 2:
            model = OneVsRestClassifier(
                _build_baseline_classifier("classification", estimator=est_key, seed=seed, max_iter=2000)
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            proba = None
            try:
                proba = model.predict_proba(x_test)
            except Exception:
                pass
        else:
            model = _build_baseline_classifier(task_type, estimator=est_key, seed=seed, max_iter=2000)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            proba = model.predict_proba(x_test) if task_type == "classification" and hasattr(model, "predict_proba") else None

        metrics = evaluate_predictions(y_test, pred, proba=proba, task_type=task_type)
        score = primary_score(metrics, task_type)
        results["estimators"][est] = {"metrics": metrics, "primary_score": score}
        logger.info("leakage baseline [%s]: primary=%.4f", est, score)

    (out_dir / "leakage_baselines.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results
