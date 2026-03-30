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

from src.embeddings.aggregation import aggregate_layer
from src.embeddings.extractor import EmbeddingExtractor
from src.experiment_logging import log_layer_scores, log_ruler
from src.probing import classification_y_array, train_probes_by_layer

logger = logging.getLogger(__name__)


def extract_all_layers(
    embedding_output: dict[str, Any],
    level: str = "text",
    strategy: str = "mean",
    tokenizer=None,
    texts: list[str] | None = None,
) -> list[Any]:
    hidden_states = embedding_output["hidden_states"]
    mask = embedding_output["attention_mask"]
    layer_outputs: list[Any] = []
    for layer_tensor in hidden_states:
        aggregated = aggregate_layer(
            hidden=layer_tensor,
            mask=mask,
            level=level,
            tokenizer=tokenizer,
            texts=texts,
            strategy=strategy,
        )
        layer_outputs.append(aggregated)
    return layer_outputs


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


def intervention_with_decomposition(
    layer_outputs: list[Any],
    labels: list[int | float],
    task_type: str,
    decomposition: dict[str, Any],
    interventions: list[str],
    n_drop: int,
    *,
    probing_n_jobs: int = 1,
) -> dict[str, Any]:
    x = _layer_to_sample_matrix(layer_outputs[-1])
    out: dict[str, Any] = {}

    if "pca" in interventions and "pca" in decomposition:
        pca_components = decomposition["pca"]["components"]
        x_int = _project_out_directions(x, pca_components, n_drop=n_drop)
        out["pca"] = train_probes_by_layer(
            [x_int], labels, task_type=task_type, n_jobs=probing_n_jobs
        )

    if "probe_directions" in interventions and "probe_directions" in decomposition:
        dirs = decomposition["probe_directions"]["components"]
        x_int = _project_out_directions(x, dirs, n_drop=n_drop)
        out["probe_directions"] = train_probes_by_layer(
            [x_int], labels, task_type=task_type, n_jobs=probing_n_jobs
        )

    if "null_space" in interventions and "probe_directions" in decomposition:
        dirs = decomposition["probe_directions"]["components"]
        x_null = _null_space_remove(x, dirs)
        out["null_space"] = train_probes_by_layer(
            [x_null], labels, task_type=task_type, n_jobs=probing_n_jobs
        )

    return out


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


def _log_decomposition_summary(decomposition: dict[str, Any]) -> None:
    if "pca" in decomposition:
        var = decomposition["pca"]["explained_variance_ratio"]
        logger.info(
            "PCA: components=%s, explained variance top-5=%s",
            int(decomposition["pca"]["components"].shape[0]),
            ", ".join(f"{float(v):.4f}" for v in var[:5]),
        )
    if "probe_directions" in decomposition:
        logger.info("Linear probe directions: n=%s", decomposition["probe_directions"]["n_directions"])


def _log_intervention_scores(intervention_scores: dict[str, Any], metric: str) -> None:
    for method, scores in intervention_scores.items():
        if scores:
            log_ruler(logger, width=56)
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
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    logger.info(
        "pipeline: level=%s strategy=%s probing=%s batch_size=%s max_length=%s -> %s",
        level,
        strategy,
        probing_task_type,
        batch_size,
        max_length,
        out_dir.resolve(),
    )

    extractor = EmbeddingExtractor(model=model, tokenizer=tokenizer, max_length=max_length)
    # В encode сразу агрегируем embeddings по уровню (text/sentence/token),
    # чтобы не собирать гигантские hidden_states (N, seq_len, hidden) для всех слоёв.
    enc = extractor.encode(texts=texts, batch_size=batch_size, level=level, strategy=strategy)
    layer_outputs = enc["layer_outputs"]
    logger.info("aggregated %s layer tensors (level=%s)", len(layer_outputs), level)

    embedding_paths = save_layer_embeddings(layer_outputs, output_dir=output_dir, prefix=f"{level}_{strategy}")
    logger.info("saved %s embedding files under %s", len(embedding_paths), out_dir.resolve())

    multilabel = probing_task_type == "classification" and bool(labels) and isinstance(labels[0], (list, tuple))
    if probing_task_type == "classification":
        metric = "f1_macro" if multilabel else "accuracy"
    else:
        metric = "neg_mse"
    layer_matrices = _layer_outputs_to_matrices(layer_outputs)
    logger.info("Probing: linear probes by layer (%s)", probing_task_type)
    probing_results = train_probes_by_layer(
        layer_matrices, labels, task_type=probing_task_type, n_jobs=probing_n_jobs
    )
    if probing_results:
        log_layer_scores(
            logger,
            "Probing results",
            {str(k): float(v) for k, v in probing_results.items()},
            metric=metric,
        )

    decomposition_methods = decomposition_methods or ["pca", "probe_directions", "null_space"]
    intervention_methods = intervention_methods or ["pca", "probe_directions", "null_space"]
    logger.info("Decomposition methods: %s", decomposition_methods)
    decomposition = decompose_embeddings(
        layer_outputs=layer_outputs,
        labels=labels,
        task_type=probing_task_type,
        methods=decomposition_methods,
        pca_components=pca_components,
        seed=seed,
        probing_n_jobs=probing_n_jobs,
    )
    _log_decomposition_summary(decomposition)

    logger.info("Interventions: %s", intervention_methods)
    intervention_scores = intervention_with_decomposition(
        layer_outputs=layer_outputs,
        labels=labels,
        task_type=probing_task_type,
        decomposition=decomposition,
        interventions=intervention_methods,
        n_drop=drop_components,
        probing_n_jobs=probing_n_jobs,
    )
    _log_intervention_scores(intervention_scores, metric=metric)

    artifacts = {
        "embedding_paths": embedding_paths,
        "probing": probing_results,
        "decomposition": _serialize_decomposition(decomposition),
        "intervention": intervention_scores,
    }
    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["summary_path"] = str(summary_path)
    logger.info("wrote summary %s", summary_path.resolve())
    return artifacts
