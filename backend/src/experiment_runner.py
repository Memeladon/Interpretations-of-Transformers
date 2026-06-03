"""Общая логика полного эксперимента (fine-tuning и анализ)"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from src.config_resolve import model_name_for_family
from src.data_pipeline import ensure_processed_tracks, load_processed_records
from src.data_schema import probing_labels_for_track
from src.data_splits import load_partition, split_manifest_path
from src.embeddings.pipeline import run_embedding_pipeline
from src.finetuning import finetune_classifier
from src.language_models import load_language_model
from src.paths import ProjectPaths

log = logging.getLogger("experiment_runner")


def enabled_models(model_flags: dict[str, bool]) -> list[str]:
    return [name for name, enabled in model_flags.items() if enabled]


def runtime_params_for_family(cfg: dict, family: str) -> tuple[int, int]:
    batch_size = int(cfg["batch_size"])
    max_length = int(cfg["max_length"])
    overrides = (cfg.get("runtime_overrides") or {}).get(family, {})
    if "batch_size" in overrides:
        batch_size = int(overrides["batch_size"])
    if "max_length" in overrides:
        max_length = int(overrides["max_length"])
    return batch_size, max_length


def resolve_runtime(cfg: dict) -> dict[str, Any]:
    rt = cfg.get("runtime") or {}
    cpu = rt.get("cpu_threads", 1)
    auto_cpu = cpu is None or int(cpu) <= 0
    if auto_cpu:
        cpu = max(1, (os.cpu_count() or 1))
        if os.name == "nt" and not torch.cuda.is_available():
            log.warning(
                "Windows CPU: runtime.cpu_threads auto → 1; для скорости — CUDA или явный cpu_threads."
            )
            cpu = 1
    else:
        cpu = max(1, int(cpu))
    interop = rt.get("torch_interop_threads")
    if interop is None:
        interop = min(2, cpu)
    else:
        interop = max(1, int(interop))
    return {
        "cpu_threads": cpu,
        "torch_interop_threads": interop,
        "probing_n_jobs": int(rt.get("probing_n_jobs", 1)),
        "tokenizers_parallelism": bool(rt.get("tokenizers_parallelism", False)),
    }


def load_track_records_from_disk(cfg: dict[str, Any], paths: ProjectPaths) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    tracks_cfg = cfg.get("tracks") or {}
    processed_root = Path(cfg.get("artifacts", {}).get("data_processed", paths.data_processed))
    for track, track_cfg in tracks_cfg.items():
        if not track_cfg.get("enabled", False):
            continue
        processed = processed_root / f"{track}_records.jsonl"
        if not processed.exists():
            raise FileNotFoundError(f"Нет {processed}. Запустите prepare_data.")
        out[track] = load_processed_records(processed)
    return out


def _split_indices_for_task(cfg: dict[str, Any], track: str, task_name: str, paths: ProjectPaths) -> tuple[list[int], list[int]] | tuple[None, None]:
    if not cfg.get("probing", {}).get("use_predefined_splits", True):
        return None, None
    manifest = split_manifest_path(cfg, track=track, task_name=task_name, paths=paths)
    if not manifest.exists():
        return None, None
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    idx = data.get("indices") or {}
    train_idx = idx.get("train")
    test_idx = idx.get("test")
    if train_idx is None or test_idx is None:
        return None, None
    return train_idx, test_idx


def run_track_analysis(
    *,
    track_name: str,
    task_names: list[str],
    records: list[dict],
    models: list[str],
    cfg: dict,
    output_root: Path,
    finetuned_by_family: dict[str, Path | None] | None = None,
    probing_n_jobs: int = 1,
) -> None:
    group_rows = [r for r in records if r["task_name"] in task_names]
    if not group_rows:
        log.warning("Track %s has no records for tasks=%s", track_name, task_names)
        return

    by_task: dict[str, list[dict]] = {}
    for row in group_rows:
        by_task.setdefault(row["task_name"], []).append(row)

    paths = ProjectPaths.from_root()
    probing_cfg = cfg.get("probing") or {}
    norm_scaler = (cfg.get("normalization") or {}).get("probing_scaler", "standard")

    for task_name, rows in by_task.items():
        task_type, labels = probing_labels_for_track(rows, track_name)
        texts = [r["text"] for r in rows]
        # Pair tasks (semantic similarity/paraphrase) должны передавать обе стороны.
        # Для non-pair задач здесь будет None.
        text_pairs: list[str] | None = None
        if any(r.get("text_pair") is not None for r in rows):
            text_pairs = [str(r.get("text_pair") or "") for r in rows]
        train_idx, test_idx = _split_indices_for_task(cfg, track_name, task_name, paths)
        if train_idx is not None:
            import json

            manifest_data = json.loads(
                split_manifest_path(cfg, track=track_name, task_name=task_name, paths=paths).read_text(
                    encoding="utf-8"
                )
            )
            coverage = (manifest_data.get("coverage_validation") or {}).get("ok")
            log.info(
                "Task %s: n=%s, predefined splits (train=%s, test=%s), coverage_ok=%s",
                task_name,
                len(texts),
                len(train_idx),
                len(test_idx),
                coverage,
            )
            if coverage is False and cfg.get("splits", {}).get("fail_on_coverage_error", True):
                raise RuntimeError(
                    f"Split coverage invalid for {track_name}/{task_name}. Re-run build_splits."
                )
        else:
            log.info("Task %s: n=%s, task_type=%s (random split in probing)", task_name, len(texts), task_type)

        for family in models:
            run_batch_size, run_max_length = runtime_params_for_family(cfg, family)
            ckpt = None
            if finetuned_by_family is not None:
                ckpt = finetuned_by_family.get(family)
            if ckpt is None and track_name in ("tone", "style"):
                log.warning(
                    "No finetuned checkpoint for %s/%s — base backbone.",
                    family,
                    track_name,
                )
            model_name = model_name_for_family(cfg, family)
            device_cfg = cfg.get("device", "auto")
            if device_cfg == "auto":
                device_str = None
            else:
                device_str = device_cfg
            model, tokenizer, device = load_language_model(
                family,
                model_name=model_name,
                device=device_str,
                finetuned_checkpoint=ckpt,
            )

            pair_task = text_pairs is not None
            for level in cfg["levels"]:
                if pair_task and level == "sentence":
                    log.info(
                        "Task %s: skip level=sentence for pair input (equivalent to text-level pooling)",
                        task_name,
                    )
                    continue
                level_out_dir = output_root / "analysis" / track_name / task_name / family / level
                result = run_embedding_pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    text_pairs=text_pairs,
                    labels=labels,
                    output_dir=level_out_dir,
                    probing_task_type=task_type,
                    level=level,
                    strategy=cfg["text_strategy"],
                    batch_size=run_batch_size,
                    max_length=run_max_length,
                    seed=cfg["seed"],
                    decomposition_methods=cfg["decomposition"]["enabled_methods"],
                    intervention_methods=cfg["decomposition"]["interventions"],
                    pca_components=cfg["decomposition"]["pca_components"],
                    drop_components=cfg["decomposition"]["drop_components"],
                    probing_n_jobs=probing_n_jobs,
                    probing_test_size=float(probing_cfg.get("test_size", 0.2)),
                    probing_max_iter=int(probing_cfg.get("max_iter", 2000)),
                    probing_scaler=norm_scaler,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    layers_cfg=cfg.get("layers"),
                    probing_estimator=str(probing_cfg.get("estimator", "logistic")),
                    fail_on_single_class_test=bool(probing_cfg.get("fail_on_single_class_test", True)),
                    run_baselines=bool((cfg.get("baselines") or {}).get("enabled", True)),
                    baseline_cfg=cfg.get("baselines"),
                    records_for_baselines=rows,
                    embedding_norm_protocol=(cfg.get("normalization") or {}).get("embeddings"),
                )
                leak = result.get("leakage_comparison") or {}
                for w in leak.get("warnings") or []:
                    log.warning(w)
                probe = result.get("probing") or {}
                if probe:
                    best = max(probe, key=lambda k: probe[k])
                    log.info("[%s/%s] best layer=%s score=%.4f", family, level, best, float(probe[best]))

            log.info("Device: %s", device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_finetuning(
    cfg: dict[str, Any],
    *,
    tone_records: list[dict],
    style_records: list[dict],
    models: list[str],
    fin_out: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    ft_cfg = cfg.get("finetuning") or {}
    seed = int(cfg["seed"])
    tone_tasks = cfg["tracks"]["tone"].get("tasks", [])
    style_tasks = cfg["tracks"]["style"].get("tasks", [])
    paths = ProjectPaths.from_root()

    def _finetune_rows(track: str, records: list[dict], tasks: list[str]) -> list[dict]:
        rows = [r for r in records if r.get("task_name") in tasks]
        if not rows or not tasks:
            return rows
        task = tasks[0]
        try:
            return load_partition(cfg, track=track, task_name=task, partition="train", paths=paths)
        except FileNotFoundError:
            return rows

    tone_train = _finetune_rows("tone", tone_records, tone_tasks)
    style_train = _finetune_rows("style", style_records, style_tasks)
    tone_ckpts: dict[str, Path] = {}
    style_ckpts: dict[str, Path] = {}

    for family in models:
        base = model_name_for_family(cfg, family)
        if tone_train:
            tone_ckpts[family] = finetune_classifier(
                base_model_name=base,
                train_texts=[r["text"] for r in tone_train],
                train_labels=[int(r["label"]) for r in tone_train],
                output_dir=fin_out / family / "tone",
                problem_type="single_label",
                seed=seed,
                max_length=cfg["max_length"],
                learning_rate=float(ft_cfg.get("learning_rate", 2e-5)),
                weight_decay=float(ft_cfg.get("weight_decay", 0.01)),
                num_train_epochs=float(ft_cfg.get("epochs", 2)),
                train_batch_size=int(ft_cfg.get("train_batch_size", 4)),
                eval_batch_size=int(ft_cfg.get("eval_batch_size", 8)),
                skip_if_exists=bool(ft_cfg.get("skip_if_exists", True)),
                dataloader_num_workers=ft_cfg.get("dataloader_num_workers"),
                fp16=ft_cfg.get("fp16"),
                bf16=bool(ft_cfg.get("bf16", False)),
                gradient_accumulation_steps=int(ft_cfg.get("gradient_accumulation_steps", 1)),
            )
        if style_train:
            style_ckpts[family] = finetune_classifier(
                base_model_name=base,
                train_texts=[r["text"] for r in style_train],
                train_labels=[
                    list(r.get("labels") or r.get("y_style") or []) for r in style_train
                ],
                output_dir=fin_out / family / "style",
                problem_type="multi_label",
                seed=seed,
                max_length=cfg["max_length"],
                learning_rate=float(ft_cfg.get("learning_rate", 2e-5)),
                weight_decay=float(ft_cfg.get("weight_decay", 0.01)),
                num_train_epochs=float(ft_cfg.get("epochs", 2)),
                train_batch_size=int(ft_cfg.get("train_batch_size", 4)),
                eval_batch_size=int(ft_cfg.get("eval_batch_size", 8)),
                skip_if_exists=bool(ft_cfg.get("skip_if_exists", True)),
                dataloader_num_workers=ft_cfg.get("dataloader_num_workers"),
                fp16=ft_cfg.get("fp16"),
                bf16=bool(ft_cfg.get("bf16", False)),
                gradient_accumulation_steps=int(ft_cfg.get("gradient_accumulation_steps", 1)),
            )
    return tone_ckpts, style_ckpts


def run_full_analysis(
    cfg: dict[str, Any],
    *,
    paths: ProjectPaths,
    artifact_root: Path | None = None,
    use_processed_data: bool = True,
    prepare_data_if_missing: bool = True,
) -> Path:
    """Полный pipeline: данные → fine-tune → encode/probe/intervene."""
    paths.ensure_dirs()
    if use_processed_data:
        if prepare_data_if_missing:
            ensure_processed_tracks(cfg, paths=paths)
        records_by_track = load_track_records_from_disk(cfg, paths)
    else:
        ensure_processed_tracks(cfg, paths=paths, force=True)
        records_by_track = load_track_records_from_disk(cfg, paths)

    rt = resolve_runtime(cfg)
    models = enabled_models(cfg["models"])
    if not models:
        raise ValueError("All models are disabled in config")

    output_root = artifact_root or paths.artifacts
    ft_cfg = cfg.get("finetuning") or {}
    fin_out = Path(ft_cfg.get("output_dir", str(paths.artifacts / "finetuned")))
    tone_ckpts: dict[str, Path] = {}
    style_ckpts: dict[str, Path] = {}

    if ft_cfg.get("enabled", False):
        tone_ckpts, style_ckpts = run_finetuning(
            cfg,
            tone_records=records_by_track.get("tone", []),
            style_records=records_by_track.get("style", []),
            models=models,
            fin_out=fin_out,
        )

    if cfg["tracks"]["semantic"].get("enabled", False):
        run_track_analysis(
            track_name="semantic",
            task_names=cfg["tracks"]["semantic"]["tasks"],
            records=records_by_track.get("semantic", []),
            models=models,
            cfg=cfg,
            output_root=output_root,
            finetuned_by_family=None,
            probing_n_jobs=rt["probing_n_jobs"],
        )
    if cfg["tracks"]["tone"].get("enabled", False):
        run_track_analysis(
            track_name="tone",
            task_names=cfg["tracks"]["tone"]["tasks"],
            records=records_by_track.get("tone", []),
            models=models,
            cfg=cfg,
            output_root=output_root,
            finetuned_by_family={m: tone_ckpts.get(m) for m in models},
            probing_n_jobs=rt["probing_n_jobs"],
        )
    if cfg["tracks"]["style"].get("enabled", False):
        run_track_analysis(
            track_name="style",
            task_names=cfg["tracks"]["style"]["tasks"],
            records=records_by_track.get("style", []),
            models=models,
            cfg=cfg,
            output_root=output_root,
            finetuned_by_family={m: style_ckpts.get(m) for m in models},
            probing_n_jobs=rt["probing_n_jobs"],
        )
    return output_root
