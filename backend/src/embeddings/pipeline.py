from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.embeddings.extractor import EmbeddingExtractor
from src.experiment_logging import log_layer_scores, log_ruler
from src.probing.layer_probe import classification_y_array, train_probes_by_layer
from src.probing.metrics import evaluate_predictions, primary_score
from src.probing.models import build_probe_pipeline
from src.baselines.compare import compare_probing_to_baselines, save_comparison_report
from src.baselines.leakage_baselines import run_leakage_baselines
from src.concept_directions.builders import build_concept_directions
from src.concept_directions.stats import direction_statistics
from src.embeddings.normalization import NormalizationProtocol

logger = logging.getLogger(__name__)


def save_layer_embeddings(layer_outputs: list[Any], output_dir: str | Path, prefix: str) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for layer_idx, tensor_or_list in enumerate(layer_outputs):
        file_path = out_dir / f"{prefix}_layer_{layer_idx}.npy"
        if isinstance(tensor_or_list, list):
            arr = np.array([x.numpy() for x in tensor_or_list], dtype=object)
        else:
            arr = tensor_or_list.numpy()
        np.save(file_path, arr, allow_pickle=True)
        saved_paths.append(str(file_path))
    return saved_paths


def _layer_to_sample_matrix(layer_output: Any) -> np.ndarray:
    if isinstance(layer_output, np.ndarray):
        return layer_output
    if hasattr(layer_output, "numpy"):
        return layer_output.numpy()

    rows = []
    for sample in layer_output:
        sample_np = sample.numpy()
        if sample_np.ndim == 1:
            rows.append(sample_np)
        else:
            rows.append(sample_np.mean(axis=0))
    return np.stack(rows, axis=0)


def _build_probe(task_type: str, seed: int, *, n_jobs: int = 1) -> Pipeline:
    if task_type == "classification":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed, n_jobs=n_jobs)),
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


def _fit_probe_directions(
    x: np.ndarray, y: np.ndarray, task_type: str, seed: int, *, n_jobs: int = 1
) -> np.ndarray:
    if task_type == "classification" and y.ndim == 2:
        model = OneVsRestClassifier(_build_probe(task_type="classification", seed=seed, n_jobs=n_jobs))
        model.fit(x, y)
        direction_rows: list[np.ndarray] = []
        for est in model.estimators_:
            scaler = est.named_steps["scale"]
            clf = est.named_steps["clf"]
            coef = clf.coef_
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            scale = scaler.scale_.copy()
            scale[scale == 0] = 1.0
            dirs = coef / scale[None, :]
            for row in dirs:
                direction_rows.append(row / (np.linalg.norm(row) + 1e-12))
        return np.stack(direction_rows, axis=0)

    model = _build_probe(task_type=task_type, seed=seed, n_jobs=n_jobs)
    model.fit(x, y)
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]

    coef = clf.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0
    directions = coef / scale[None, :]

    norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    return directions / norms


def _null_space_remove(x: np.ndarray, directions: np.ndarray) -> np.ndarray:
    w = directions
    ww_t = w @ w.T
    pinv = np.linalg.pinv(ww_t)
    p = np.eye(x.shape[1]) - (w.T @ pinv @ w)
    return x @ p


def _project_out_directions(x: np.ndarray, directions: np.ndarray, n_drop: int = 1) -> np.ndarray:
    basis = directions[:n_drop]
    projection = x @ basis.T @ basis
    return x - projection


def decompose_embeddings(
    layer_outputs: list[Any],
    labels: list[int | float],
    task_type: str,
    methods: list[str],
    pca_components: int,
    seed: int,
    *,
    probing_n_jobs: int = 1,
) -> dict[str, Any]:
    x = _layer_to_sample_matrix(layer_outputs[-1])
    if task_type == "classification":
        y = classification_y_array(list(labels))
    else:
        y = np.asarray(labels, dtype=np.float64)
    out: dict[str, Any] = {}

    if "pca" in methods:
        pca = PCA(n_components=min(pca_components, x.shape[1]))
        proj = pca.fit_transform(x)
        out["pca"] = {
            "components": pca.components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "projections": proj,
        }

    if "probe_directions" in methods or "null_space" in methods:
        directions = _fit_probe_directions(
            x=x, y=y, task_type=task_type, seed=seed, n_jobs=probing_n_jobs
        )
        out["probe_directions"] = {
            "components": directions,
            "n_directions": int(directions.shape[0]),
        }

    return out


def _cosine_drift(x_orig: np.ndarray, x_new: np.ndarray, indices: list[int] | None = None) -> float:
    if indices is not None:
        x_orig = x_orig[indices]
        x_new = x_new[indices]
    orig_norm = np.linalg.norm(x_orig, axis=1, keepdims=True) + 1e-12
    new_norm = np.linalg.norm(x_new, axis=1, keepdims=True) + 1e-12
    cos = np.sum((x_orig / orig_norm) * (x_new / new_norm), axis=1)
    return float(np.mean(1.0 - cos))


def _probe_eval_on_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    task_type: str,
    train_indices: list[int],
    test_indices: list[int],
    seed: int,
    estimator: str,
    scaler: str,
    n_jobs: int,
) -> dict[str, Any]:
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    multilabel = task_type == "classification" and y.ndim == 2
    if multilabel:
        base = build_probe_pipeline(
            "classification",
            estimator=estimator,
            seed=seed,
            max_iter=2000,
            scaler=scaler,
            n_jobs=n_jobs,
        )
        model = OneVsRestClassifier(base)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = None
    else:
        model = build_probe_pipeline(
            task_type,
            estimator=estimator,
            seed=seed,
            max_iter=2000,
            scaler=scaler,
            n_jobs=n_jobs,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test) if task_type == "classification" else None
    metrics = evaluate_predictions(y_test, pred, proba=proba, task_type=task_type)
    return {
        "metrics": metrics,
        "primary_score": primary_score(metrics, task_type),
    }


def _evaluate_intervention(
    x_orig: np.ndarray,
    x_int: np.ndarray,
    labels: list[int | float] | list[list[int]],
    *,
    task_type: str,
    train_indices: list[int] | None,
    test_indices: list[int] | None,
    seed: int,
    estimator: str,
    scaler: str,
    n_jobs: int,
) -> dict[str, Any]:
    if task_type == "classification":
        y = classification_y_array(list(labels))
    else:
        y = np.asarray(labels, dtype=np.float64)

    if train_indices is None or test_indices is None:
        scores = train_probes_by_layer(
            [x_int],
            labels,
            task_type=task_type,
            n_jobs=n_jobs,
            estimator=estimator,
            seed=seed,
        )
        return {"layer_scores": scores}

    before = _probe_eval_on_split(
        x_orig,
        y,
        task_type=task_type,
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
        estimator=estimator,
        scaler=scaler,
        n_jobs=n_jobs,
    )
    after = _probe_eval_on_split(
        x_int,
        y,
        task_type=task_type,
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
        estimator=estimator,
        scaler=scaler,
        n_jobs=n_jobs,
    )
    drift = _cosine_drift(x_orig, x_int, test_indices)
    target_drop = before["primary_score"] - after["primary_score"]
    denom = abs(before["primary_score"]) + 1e-12
    return {
        "before": before,
        "after": after,
        "target_score_drop": float(target_drop),
        "relative_target_drop": float(target_drop / denom),
        "cosine_drift_mean": drift,
        "selectivity_score": float(target_drop / (target_drop + drift + 1e-12)),
        "layer_scores": {"0": after["primary_score"]},
    }


def intervention_with_decomposition(
    layer_outputs: list[Any],
    labels: list[int | float],
    task_type: str,
    decomposition: dict[str, Any],
    interventions: list[str],
    n_drop: int,
    *,
    probing_n_jobs: int = 1,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    seed: int = 42,
    probing_scaler: str = "standard",
) -> dict[str, Any]:
    x = _layer_to_sample_matrix(layer_outputs[-1])
    out: dict[str, Any] = {}
    estimator = "ridge" if task_type == "regression" else "logistic"

    if "pca" in interventions and "pca" in decomposition:
        pca_components = decomposition["pca"]["components"]
        x_int = _project_out_directions(x, pca_components, n_drop=n_drop)
        out["pca"] = _evaluate_intervention(
            x,
            x_int,
            labels,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
            seed=seed,
            estimator=estimator,
            scaler=probing_scaler,
            n_jobs=probing_n_jobs,
        )

    if "probe_directions" in interventions and "probe_directions" in decomposition:
        dirs = decomposition["probe_directions"]["components"]
        x_int = _project_out_directions(x, dirs, n_drop=n_drop)
        out["probe_directions"] = _evaluate_intervention(
            x,
            x_int,
            labels,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
            seed=seed,
            estimator=estimator,
            scaler=probing_scaler,
            n_jobs=probing_n_jobs,
        )

    if "null_space" in interventions and "probe_directions" in decomposition:
        dirs = decomposition["probe_directions"]["components"]
        x_null = _null_space_remove(x, dirs)
        out["null_space"] = _evaluate_intervention(
            x,
            x_null,
            labels,
            task_type=task_type,
            train_indices=train_indices,
            test_indices=test_indices,
            seed=seed,
            estimator=estimator,
            scaler=probing_scaler,
            n_jobs=probing_n_jobs,
        )

    return out


def save_concept_directions(decomposition: dict[str, Any], output_dir: Path) -> Path | None:
    """Сохранение concept directions (PCA + probe) для повторного использования."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if "pca" in decomposition:
        payload["pca"] = {
            "components": decomposition["pca"]["components"].tolist(),
            "explained_variance_ratio": decomposition["pca"]["explained_variance_ratio"].tolist(),
        }
    if "probe_directions" in decomposition:
        payload["probe_directions"] = {
            "components": decomposition["probe_directions"]["components"].tolist(),
            "n_directions": decomposition["probe_directions"]["n_directions"],
        }
    if not payload:
        return None
    dest = out_dir / "concept_directions.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def run_probing_stage(
    layer_outputs: list[Any],
    labels: list[int | float] | list[list[int]],
    *,
    probing_task_type: str = "classification",
    probing_n_jobs: int = 1,
    seed: int = 42,
    test_size: float = 0.2,
    max_iter: int = 2000,
    scaler: str = "standard",
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    estimator: str = "logistic",
    output_dir: Path | None = None,
    strict_test_coverage: bool = True,
) -> dict[str, Any]:
    multilabel = probing_task_type == "classification" and bool(labels) and isinstance(labels[0], (list, tuple))
    if probing_task_type == "regression" and estimator in {"logistic", "svm"}:
        estimator = "ridge"
    layer_matrices = _layer_outputs_to_matrices(layer_outputs)
    probing_results = train_probes_by_layer(
        layer_matrices,
        labels,
        task_type=probing_task_type,
        seed=seed,
        n_jobs=probing_n_jobs,
        test_size=test_size,
        max_iter=max_iter,
        scaler=scaler,
        train_indices=train_indices,
        test_indices=test_indices,
        estimator=estimator,
        output_dir=output_dir,
        strict_test_coverage=strict_test_coverage,
    )
    metric = "f1_macro" if multilabel else ("accuracy" if probing_task_type == "classification" else "neg_mse")
    if probing_results:
        log_layer_scores(
            logger,
            "Probing results",
            {str(k): float(v) for k, v in probing_results.items()},
            metric=metric,
        )
    return {"probing": probing_results, "metric": metric}


def run_concept_and_intervention_stage(
    layer_outputs: list[Any],
    labels: list[int | float] | list[list[int]],
    *,
    output_dir: str | Path,
    probing_task_type: str = "classification",
    decomposition_methods: list[str] | None = None,
    intervention_methods: list[str] | None = None,
    pca_components: int = 8,
    drop_components: int = 1,
    seed: int = 42,
    probing_n_jobs: int = 1,
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    probing_scaler: str = "standard",
) -> dict[str, Any]:
    """Concept directions + representation interventions."""
    decomposition_methods = decomposition_methods or ["pca", "probe_directions", "null_space"]
    intervention_methods = intervention_methods or ["pca", "probe_directions", "null_space"]
    decomposition = decompose_embeddings(
        layer_outputs=layer_outputs,
        labels=labels,
        task_type=probing_task_type,
        methods=decomposition_methods,
        pca_components=pca_components,
        seed=seed,
        probing_n_jobs=probing_n_jobs,
    )
    x = _layer_to_sample_matrix(layer_outputs[-1])
    _log_decomposition_summary(
        decomposition,
        n_samples=x.shape[0],
        embedding_dim=x.shape[1],
    )
    out_path = Path(output_dir)
    save_concept_directions(decomposition, out_path)
    concept_cfg_methods = decomposition_methods
    built = build_concept_directions(
        x,
        list(labels),
        methods=[
            m if m != "probe_directions" else "probe_weight"
            for m in concept_cfg_methods
        ]
        + ["mean_difference", "class_separation"],
        pca_components=pca_components,
    )
    stats_report: dict[str, Any] = {}
    for name, payload in built.items():
        dirs = np.array(payload["components"])
        stats_report[name] = direction_statistics(dirs, x)
    (out_path / "concept_direction_stats.json").write_text(
        json.dumps(stats_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    multilabel = probing_task_type == "classification" and bool(labels) and isinstance(labels[0], (list, tuple))
    metric = "f1_macro" if multilabel else ("accuracy" if probing_task_type == "classification" else "neg_mse")
    intervention_scores = intervention_with_decomposition(
        layer_outputs=layer_outputs,
        labels=labels,
        task_type=probing_task_type,
        decomposition=decomposition,
        interventions=intervention_methods,
        n_drop=drop_components,
        probing_n_jobs=probing_n_jobs,
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
        probing_scaler=probing_scaler,
    )
    (out_path / "intervention_eval.json").write_text(
        json.dumps(intervention_scores, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _log_intervention_scores(intervention_scores, metric=metric)
    return {
        "decomposition": _serialize_decomposition(decomposition),
        "intervention": intervention_scores,
    }


def _serialize_decomposition(decomposition: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    if "pca" in decomposition:
        serialized["pca"] = {
            "explained_variance_ratio": decomposition["pca"]["explained_variance_ratio"].tolist(),
            "n_components": int(decomposition["pca"]["components"].shape[0]),
        }
    if "probe_directions" in decomposition:
        serialized["probe_directions"] = {
            "n_directions": int(decomposition["probe_directions"]["n_directions"]),
        }
    return serialized


def _layer_outputs_to_matrices(layer_outputs: list[Any]) -> list[np.ndarray]:
    return [_layer_to_sample_matrix(layer_output) for layer_output in layer_outputs]


def filter_layer_outputs(layer_outputs: list[Any], layers_cfg: dict[str, Any] | None) -> list[Any]:
    if not layers_cfg:
        return layer_outputs
    mode = layers_cfg.get("mode", "all")
    if mode == "all":
        return layer_outputs
    indices = layers_cfg.get("indices")
    if indices is not None:
        return [layer_outputs[int(i)] for i in indices]
    last_n = layers_cfg.get("last_n")
    if last_n is not None:
        n = int(last_n)
        return layer_outputs[-n:] if n > 0 else layer_outputs
    return layer_outputs


def _log_decomposition_summary(decomposition: dict[str, Any], *, n_samples: int, embedding_dim: int) -> None:
    if "pca" in decomposition:
        var = decomposition["pca"]["explained_variance_ratio"]
        logger.info(
            "PCA: n_samples=%s dim=%s components=%s, explained variance top-5=%s",
            n_samples,
            embedding_dim,
            int(decomposition["pca"]["components"].shape[0]),
            ", ".join(f"{float(v):.4f}" for v in var[:5]),
        )
    if "probe_directions" in decomposition:
        logger.info("Linear probe directions: n=%s", decomposition["probe_directions"]["n_directions"])


def _log_intervention_scores(intervention_scores: dict[str, Any], metric: str) -> None:
    for method, payload in intervention_scores.items():
        if not payload:
            continue
        log_ruler(logger, width=56)
        if "before" in payload and "after" in payload:
            before = payload["before"]["primary_score"]
            after = payload["after"]["primary_score"]
            logger.info(
                "Intervention %s: before=%.4f after=%.4f drop=%.4f rel_drop=%.2f%% "
                "cosine_drift=%.4f selectivity=%.4f",
                method,
                before,
                after,
                payload.get("target_score_drop", before - after),
                100.0 * float(payload.get("relative_target_drop", 0.0)),
                float(payload.get("cosine_drift_mean", 0.0)),
                float(payload.get("selectivity_score", 0.0)),
            )
            after_metrics = payload["after"].get("metrics", {})
            if "f1_macro" in after_metrics:
                logger.info("  after f1_macro=%.4f f1_micro=%.4f", after_metrics["f1_macro"], after_metrics["f1_micro"])
            elif "f1" in after_metrics:
                logger.info("  after f1=%.4f accuracy=%.4f", after_metrics["f1"], after_metrics.get("accuracy", 0))
            continue
        scores = payload.get("layer_scores") or payload
        if scores:
            log_layer_scores(
                logger,
                f"After intervention ({method})",
                {str(k): float(v) for k, v in scores.items()},
                metric=metric,
            )


def run_embedding_pipeline(
    model,
    tokenizer,
    texts: list[str],
    text_pairs: list[str] | None,
    labels: list[int | float],
    output_dir: str | Path,
    probing_task_type: str = "classification",
    level: str = "text",
    strategy: str = "mean",
    batch_size: int = 8,
    max_length: int = 256,
    seed: int = 42,
    decomposition_methods: list[str] | None = None,
    intervention_methods: list[str] | None = None,
    pca_components: int = 8,
    drop_components: int = 1,
    probing_n_jobs: int = 1,
    probing_test_size: float = 0.2,
    probing_max_iter: int = 2000,
    probing_scaler: str = "standard",
    train_indices: list[int] | None = None,
    test_indices: list[int] | None = None,
    layers_cfg: dict[str, Any] | None = None,
    probing_estimator: str = "logistic",
    fail_on_single_class_test: bool = True,
    run_baselines: bool = True,
    baseline_cfg: dict[str, Any] | None = None,
    records_for_baselines: list[dict[str, Any]] | None = None,
    embedding_norm_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    import torch

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    run_diagnostics = {
        "n_samples": len(texts),
        "pair_input": text_pairs is not None,
        "level": level,
        "pooling_strategy": strategy,
        "batch_size": batch_size,
        "max_length": max_length,
        "probing_task_type": probing_task_type,
        "probing_scaler": probing_scaler,
        "probing_estimator": probing_estimator,
        "embedding_normalization": (embedding_norm_protocol or {"method": "none"}),
        "model_device": str(model_device),
        "model_dtype": str(model_dtype),
        "cuda_available": bool(torch.cuda.is_available()),
        "predefined_split": train_indices is not None and test_indices is not None,
        "train_size": len(train_indices) if train_indices else None,
        "test_size_split": len(test_indices) if test_indices else None,
    }
    logger.info(
        "Run diagnostics: n=%s level=%s pooling=%s batch=%s max_len=%s device=%s dtype=%s "
        "norm=%s scaler=%s split=fixed(%s/%s)",
        len(texts),
        level,
        strategy,
        batch_size,
        max_length,
        model_device,
        model_dtype,
        (embedding_norm_protocol or {}).get("method", "none"),
        probing_scaler,
        len(train_indices) if train_indices else "random",
        len(test_indices) if test_indices else "random",
    )

    extractor = EmbeddingExtractor(model=model, tokenizer=tokenizer, max_length=max_length)
    # В encode сразу агрегируем embeddings по уровню (text/sentence/token),
    # чтобы не собирать гигантские hidden_states (N, seq_len, hidden) для всех слоёв.
    enc = extractor.encode(
        texts=texts,
        text_pairs=text_pairs,
        batch_size=batch_size,
        level=level,
        strategy=strategy,
    )
    layer_outputs = filter_layer_outputs(enc["layer_outputs"], layers_cfg)
    run_diagnostics["n_layers"] = len(layer_outputs)
    sample_matrix = _layer_to_sample_matrix(layer_outputs[-1])
    run_diagnostics["embedding_dim"] = int(sample_matrix.shape[1])
    logger.info("aggregated %s layer tensors (level=%s)", len(layer_outputs), level)

    embedding_paths = save_layer_embeddings(layer_outputs, output_dir=output_dir, prefix=f"{level}_{strategy}")
    logger.info("saved %s embedding files under %s", len(embedding_paths), out_dir.resolve())

    logger.info("Probing: linear probes by layer (%s)", probing_task_type)
    probe_out = out_dir / "probing"
    probe_stage = run_probing_stage(
        layer_outputs,
        labels,
        probing_task_type=probing_task_type,
        probing_n_jobs=probing_n_jobs,
        seed=seed,
        test_size=probing_test_size,
        max_iter=probing_max_iter,
        scaler=probing_scaler,
        train_indices=train_indices,
        test_indices=test_indices,
        estimator=probing_estimator,
        output_dir=probe_out,
        strict_test_coverage=fail_on_single_class_test,
    )
    probing_results = probe_stage["probing"]

    norm_proto = NormalizationProtocol.from_config(embedding_norm_protocol or {"method": "none"})
    norm_proto.save(out_dir / "embedding_normalization_protocol.json")

    baseline_report = None
    if run_baselines and records_for_baselines:
        bl_cfg = baseline_cfg or {}
        baseline_results = run_leakage_baselines(
            records_for_baselines,
            labels,
            probing_task_type,
            output_dir=out_dir / "baselines",
            estimators=bl_cfg.get("estimators", ["logistic", "svm"]),
            domain_vocabulary=bl_cfg.get("domain_vocabulary"),
            seed=seed,
            test_size=probing_test_size,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        baseline_report = compare_probing_to_baselines(
            probing_results,
            baseline_results,
            tolerance=float(bl_cfg.get("tolerance", 0.05)),
        )
        save_comparison_report(baseline_report, out_dir / "leakage_comparison.json")

    logger.info("Decomposition + interventions")
    concept_stage = run_concept_and_intervention_stage(
        layer_outputs,
        labels,
        output_dir=out_dir,
        probing_task_type=probing_task_type,
        decomposition_methods=decomposition_methods,
        intervention_methods=intervention_methods,
        pca_components=pca_components,
        drop_components=drop_components,
        seed=seed,
        probing_n_jobs=probing_n_jobs,
        train_indices=train_indices,
        test_indices=test_indices,
        probing_scaler=probing_scaler,
    )
    (out_dir / "run_diagnostics.json").write_text(
        json.dumps(run_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    task_meta = {
        "task_type": probing_task_type,
        "labels": labels,
        "n_samples": len(texts),
    }
    (out_dir / "task_meta.json").write_text(
        json.dumps(task_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    artifacts = {
        "embedding_paths": embedding_paths,
        "probing": probing_results,
        "probing_profile_P_f_l": probing_results,
        "decomposition": concept_stage["decomposition"],
        "intervention": concept_stage["intervention"],
        "leakage_comparison": baseline_report,
        "normalization_protocol": norm_proto.to_dict(),
    }
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["summary_path"] = str(summary_path)
    logger.info("wrote summary %s", summary_path.resolve())
    return artifacts
