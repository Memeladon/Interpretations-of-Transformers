"""Фабрика probing-моделей: logistic, SVM, ridge, linear regression"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def resolve_probe_estimator(task_type: str, estimator: str) -> str:
    """Подбирает estimator под тип задачи (глобальный конфиг часто задаёт logistic)."""
    if task_type == "regression" and estimator in {"logistic", "svm"}:
        return "ridge"
    if task_type == "classification" and estimator in {"ridge", "linear"}:
        return "logistic"
    return estimator


def build_probe_pipeline(
    task_type: str,
    *,
    estimator: str = "logistic",
    seed: int = 42,
    max_iter: int = 2000,
    scaler: str = "standard",
    n_jobs: int = 1,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if scaler == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler != "none":
        raise ValueError(f"Unknown scaler: {scaler}")

    estimator = resolve_probe_estimator(task_type, estimator)

    if task_type == "classification":
        if estimator == "logistic":
            clf: Any = LogisticRegression(max_iter=max_iter, random_state=seed, n_jobs=n_jobs)
        elif estimator == "svm":
            clf = LinearSVC(max_iter=max_iter, random_state=seed, dual="auto")
        else:
            raise ValueError(f"Unknown classifier: {estimator}")
    elif task_type == "regression":
        if estimator == "ridge":
            clf = Ridge(random_state=seed)
        elif estimator == "linear":
            clf = LinearRegression()
        else:
            raise ValueError(f"Unknown regressor: {estimator}")
    else:
        raise ValueError(task_type)

    steps.append(("clf", clf))
    return Pipeline(steps)
