"""Нормализация декларативного конфига эксперимента"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def resolve_experiment_config(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Приводит YAML к единому resolved-словарю с полями
    """
    cfg = deepcopy(raw)

    run = cfg.get("run") or {}
    model = cfg.get("model") or {}
    tokenization = cfg.get("tokenization") or {}
    layers = cfg.get("layers") or {}
    pooling = cfg.get("pooling") or {}
    normalization = cfg.get("normalization") or {}
    probing = cfg.get("probing") or {}
    interventions = cfg.get("interventions") or cfg.get("decomposition") or {}
    artifacts = cfg.get("artifacts") or {}
    data = cfg.get("data") or {}

    # --- идентификатор запуска ---
    cfg["run_id"] = run.get("id") or cfg.get("run_id")

    # --- модель ---
    families_cfg = model.get("families") or {}
    legacy_models = cfg.get("models") or {}
    if not families_cfg and legacy_models:
        from src.language_models.loader import MODEL_REGISTRY

        families_cfg = {
            fam: {"name": MODEL_REGISTRY.get(fam, fam), "path": None, "enabled": bool(en)}
            for fam, en in legacy_models.items()
        }
    cfg["model_families"] = families_cfg
    cfg["models"] = {k: v.get("enabled", True) for k, v in families_cfg.items()} if families_cfg else legacy_models

    active = model.get("active") or model.get("active_family")
    if active:
        cfg["active_model_family"] = active

    # --- токенизация ---
    cfg["max_length"] = int(tokenization.get("max_length", cfg.get("max_length", 256)))
    cfg["tokenization"] = {
        "max_length": cfg["max_length"],
        "padding": tokenization.get("padding", "max_length"),
        "truncation": bool(tokenization.get("truncation", True)),
    }

    # --- слои ---
    cfg["layers"] = {
        "mode": layers.get("mode", "all"),
        "indices": layers.get("indices"),
        "last_n": layers.get("last_n"),
    }

    # --- pooling ---
    cfg["text_strategy"] = pooling.get("strategy", cfg.get("text_strategy", "mean"))
    cfg["levels"] = pooling.get("levels", cfg.get("levels", ["text"]))
    cfg["pooling"] = {"strategy": cfg["text_strategy"], "levels": cfg["levels"]}

    # --- нормировка ---
    cfg["normalization"] = {
        "probing_scaler": normalization.get("probing_scaler", "standard"),
        "embeddings": normalization.get(
            "embeddings",
            {"method": normalization.get("embedding_method", "none")},
        ),
    }

    # --- probing ---
    splits = cfg.get("splits") or {}
    cfg["probing"] = {
        "test_size": float(probing.get("test_size", splits.get("test_size", 0.2))),
        "n_jobs": int(probing.get("n_jobs", (cfg.get("runtime") or {}).get("probing_n_jobs", 1))),
        "max_iter": int(probing.get("max_iter", 2000)),
        "use_predefined_splits": bool(probing.get("use_predefined_splits", True)),
        "estimator": str(probing.get("estimator", "logistic")),
        "estimators": list(probing.get("estimators", ["logistic", "svm", "ridge"])),
    }

    # --- leakage baselines (поверхностные признаки) ---
    baselines = cfg.get("baselines") or {}
    cfg["baselines"] = {
        "enabled": bool(baselines.get("enabled", True)),
        "estimators": list(baselines.get("estimators", ["logistic", "svm"])),
        "tolerance": float(baselines.get("tolerance", 0.05)),
        "domain_vocabulary": list(baselines.get("domain_vocabulary") or []),
        "char_ngram_dim": int(baselines.get("char_ngram_dim", 256)),
    }

    # --- извлечение представлений ---
    extraction = cfg.get("extraction") or {}
    cfg["extraction"] = {
        "use_amp": bool(extraction.get("use_amp", False)),
        "chunk_size": extraction.get("chunk_size"),
        "cache_dir": str(extraction.get("cache_dir", "artifacts/embedding_cache")),
        "output_attentions": bool(extraction.get("output_attentions", False)),
    }

    # --- concept directions ---
    concept = cfg.get("concept_directions") or {}
    cfg["concept_directions"] = {
        "methods": list(
            concept.get(
                "methods",
                ["probe_weight", "mean_difference", "pca", "class_separation"],
            )
        ),
        "pca_components": int(concept.get("pca_components", 8)),
    }
    if "runtime" not in cfg:
        cfg["runtime"] = {}
    cfg["runtime"]["probing_n_jobs"] = cfg["probing"]["n_jobs"]

    # --- интервенции ---
    cfg["decomposition"] = {
        "pca_components": int(interventions.get("pca_components", 8)),
        "enabled_methods": list(
            interventions.get("enabled_methods", ["pca", "probe_directions", "null_space"])
        ),
        "interventions": list(
            interventions.get("methods", interventions.get("interventions", ["pca", "probe_directions", "null_space"]))
        ),
        "drop_components": int(interventions.get("drop_components", 1)),
    }

    # --- batch, seed, device ---
    cfg["batch_size"] = int(cfg.get("batch_size", 8))
    cfg["seed"] = int(cfg.get("seed", 42))
    device = cfg.get("device", "auto")
    cfg["device"] = device

    # --- пути артефактов ---
    cfg["artifacts"] = {
        "root": str(artifacts.get("root", "artifacts")),
        "data_processed": str(artifacts.get("data_processed", "data/processed")),
        "data_splits": str(artifacts.get("data_splits", "data/splits")),
        "reports": str(artifacts.get("reports", "reports")),
        "finetuned": str(artifacts.get("finetuned", "artifacts/finetuned")),
        "logs": str(artifacts.get("logs", "logs")),
        "runs": str(artifacts.get("runs", "runs")),
        "embeddings": str(artifacts.get("embeddings", "artifacts/embeddings")),
    }
    if "finetuning" in cfg and isinstance(cfg["finetuning"], dict):
        cfg["finetuning"].setdefault("output_dir", cfg["artifacts"]["finetuned"])

    # --- данные ---
    prep = data.get("preprocessing") or {}
    cfg["data"] = {
        "sources": data.get("sources") or {"default": "huggingface"},
        "preprocessing": {
            "version": str(prep.get("version", "1.0")),
            "clean_text": bool(prep.get("clean_text", True)),
            "normalize_whitespace": bool(prep.get("normalize_whitespace", True)),
            "deduplicate": bool(prep.get("deduplicate", True)),
            "filter_empty": bool(prep.get("filter_empty", True)),
            "max_text_length": int(prep.get("max_text_length", 10000)),
            "balance_classes": bool(prep.get("balance_classes", False)),
            "balance_max_per_class": prep.get("balance_max_per_class"),
        },
        "leakage": data.get("leakage")
        or {
            "enabled": True,
            "check_duplicate_text": True,
            "check_pair_id_overlap": True,
            "check_source_overlap": True,
            "fail_on_violation": True,
        },
    }
    cfg["cache_dir"] = data.get("cache_dir", cfg.get("cache_dir", ".cache/hf_datasets"))
    cfg["dataset_limit_per_source"] = data.get("limit_per_source", cfg.get("dataset_limit_per_source"))

    cfg["splits"] = {
        "test_size": float(splits.get("test_size", cfg["probing"]["test_size"])),
        "val_size": float(splits.get("val_size", 0.1)),
    }

    return cfg


def model_name_for_family(cfg: dict[str, Any], family: str) -> str:
    families = cfg.get("model_families") or {}
    entry = families.get(family) or {}
    if entry.get("path"):
        return str(entry["path"])
    if entry.get("name"):
        return str(entry["name"])
    from src.language_models.loader import MODEL_REGISTRY

    return MODEL_REGISTRY[family]
