from __future__ import annotations

import logging
import os
import sys
import faulthandler
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# До import torch: на Windows многопоточный BLAS/OpenMP + PyTorch на CPU часто даёт
# access violation в нативном коде при длинных forward (BERT). Ограничиваем BLAS.
if os.name == "nt":
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_k, "1")

import numpy as np
import torch

from src.datasets_load import load_track_datasets
from src.embeddings.pipeline import run_embedding_pipeline
from src.experiment_config import load_experiment_config
from src.experiment_logging import log_ruler, log_section, setup_experiment_logging
from src.finetuning import finetune_classifier
from src.language_models import base_model_name_for_family, load_language_model

log = logging.getLogger("run_exp")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enabled_models(model_flags: dict[str, bool]) -> list[str]:
    return [name for name, enabled in model_flags.items() if enabled]


def _runtime_params_for_family(cfg: dict, family: str) -> tuple[int, int]:
    batch_size = int(cfg["batch_size"])
    max_length = int(cfg["max_length"])
    overrides = (cfg.get("runtime_overrides") or {}).get(family, {})
    if "batch_size" in overrides:
        batch_size = int(overrides["batch_size"])
    if "max_length" in overrides:
        max_length = int(overrides["max_length"])
    return batch_size, max_length


def _resolve_runtime(cfg: dict) -> dict:
    """cpu_threads 0 или null → авто; probing_n_jobs=-1 → все ядра в sklearn."""
    rt = cfg.get("runtime") or {}
    cpu = rt.get("cpu_threads", 1)
    auto_cpu = cpu is None or int(cpu) <= 0
    if auto_cpu:
        cpu = max(1, (os.cpu_count() or 1))
        # Windows + CPU: «все ядра» в torch часто ломает нативный стек (GELU/linear) на длинном encode.
        if os.name == "nt" and not torch.cuda.is_available():
            log.warning(
                "Windows CPU: runtime.cpu_threads auto → 1 (PyTorch+OpenMP на всех ядрах даёт access violation); "
                "для скорости используйте CUDA или явно задайте cpu_threads."
            )
            cpu = 1
    else:
        cpu = max(1, int(cpu))
    interop = rt.get("torch_interop_threads")
    if interop is None:
        interop = min(2, cpu)
    else:
        interop = max(1, int(interop))
    probing_n = int(rt.get("probing_n_jobs", 1))
    tok_par = bool(rt.get("tokenizers_parallelism", False))
    out = {
        "cpu_threads": cpu,
        "torch_interop_threads": interop,
        "probing_n_jobs": probing_n,
        "tokenizers_parallelism": tok_par,
    }
    return out


def _run_track_analysis(
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

    log_section(log, f"Track: {track_name}")
    for task_name, rows in by_task.items():
        task_type = rows[0]["task_type"]
        texts = [r["text"] for r in rows]
        if task_type == "classification" and rows[0].get("label_type") == "multi_label":
            labels = [r.get("labels") or [] for r in rows]
        else:
            labels = [r["label"] for r in rows]
        log.info("Task %s: n=%s, task_type=%s", task_name, len(texts), task_type)
        log_ruler(log)

        for family in models:
            log_section(log, f"Model: {family.upper()} | track: {track_name} | task: {task_name}")
            run_batch_size, run_max_length = _runtime_params_for_family(cfg, family)
            ckpt = None
            if finetuned_by_family is not None:
                ckpt = finetuned_by_family.get(family)
            if ckpt is None and track_name in ("tone", "style"):
                log.warning(
                    "No finetuned checkpoint for %s/%s — using base backbone (expect worse alignment with the task).",
                    family,
                    track_name,
                )
            model, tokenizer, device = load_language_model(family, finetuned_checkpoint=ckpt)

            for level in cfg["levels"]:
                level_out_dir = output_root / "analysis" / track_name / task_name / family / level
                log.info("Analysis level: %s (batch_size=%s, max_length=%s)", level, run_batch_size, run_max_length)
                result = run_embedding_pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
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
                )
                probe = result.get("probing") or {}
                if probe:
                    best = max(probe, key=lambda k: probe[k])
                    log.info("[%s/%s] best layer=%s score=%.4f", family, level, best, float(probe[best]))
                log.info("summary: %s", result.get("summary_path", ""))

            log.info("Device: %s", device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    cfg_path = Path("experiments/experiment_config.json")
    cfg = load_experiment_config(cfg_path)

    output_root = Path("artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = setup_experiment_logging(output_root / "logs")
    log_section(log, "Transformer interpretation experiment")
    log.info("Config: %s", cfg_path.resolve())
    log.info("Log file: %s", log_path.resolve())
    # Пишем traceback даже для фатальных нативных падений (если процесс успеет сбросить буфер).
    crash_log = open(log_path, "a", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)
    log.info(
        "faulthandler enabled on log file (no periodic dumps: long CPU encode would look like false timeouts)."
    )

    try:
        set_seed(cfg["seed"])
        rt = _resolve_runtime(cfg)
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if rt["tokenizers_parallelism"] else "false"
        torch.set_num_threads(rt["cpu_threads"])
        torch.set_num_interop_threads(rt["torch_interop_threads"])
        log.info(
            "Runtime: torch num_threads=%s interop_threads=%s TOKENIZERS_PARALLELISM=%s probing_n_jobs=%s",
            rt["cpu_threads"],
            rt["torch_interop_threads"],
            os.environ["TOKENIZERS_PARALLELISM"],
            rt["probing_n_jobs"],
        )
        models = _enabled_models(cfg["models"])
        if not models:
            raise ValueError("All models are disabled in config: models")
        log.info("Enabled models: %s", models)

        ds_limit = cfg["dataset_limit_per_source"]
        cache_dir = cfg["cache_dir"]
        seed = cfg["seed"]

        semantic_records: list[dict] = []
        tone_records: list[dict] = []
        style_records: list[dict] = []

        log_section(log, "Stage: load datasets by track (semantic / tone / style)")
        if cfg["tracks"]["semantic"].get("enabled", False):
            semantic_records = load_track_datasets("semantic", cache_dir=cache_dir, limit_per_dataset=ds_limit, seed=seed)
            log.info("Semantic records: %s", len(semantic_records))
        if cfg["tracks"]["tone"].get("enabled", False):
            tone_records = load_track_datasets("tone", cache_dir=cache_dir, limit_per_dataset=ds_limit, seed=seed)
            log.info("Tone records: %s", len(tone_records))
        if cfg["tracks"]["style"].get("enabled", False):
            style_records = load_track_datasets("style", cache_dir=cache_dir, limit_per_dataset=ds_limit, seed=seed)
            log.info("Style records: %s", len(style_records))

        ft_cfg = cfg.get("finetuning") or {}
        fin_out = Path(ft_cfg.get("output_dir", "artifacts/finetuned"))
        tone_ckpts: dict[str, Path] = {}
        style_ckpts: dict[str, Path] = {}

        if ft_cfg.get("enabled", False):
            log.info(
                "Finetune hardware: dataloader_num_workers=%s fp16=%s bf16=%s gradient_accumulation_steps=%s",
                ft_cfg.get("dataloader_num_workers"),
                ft_cfg.get("fp16"),
                ft_cfg.get("bf16", False),
                ft_cfg.get("gradient_accumulation_steps", 1),
            )
            log_section(log, "Stage: fine-tuning (tone + style heads per model)")
            tone_tasks = cfg["tracks"]["tone"].get("tasks", [])
            style_tasks = cfg["tracks"]["style"].get("tasks", [])
            tone_train = [r for r in tone_records if r["task_name"] in tone_tasks]
            style_train = [r for r in style_records if r["task_name"] in style_tasks]

            for family in models:
                base = base_model_name_for_family(family)

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
                        train_labels=[list(r.get("labels") or []) for r in style_train],
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
        else:
            log.info("Fine-tuning disabled in config.")

        log.info("Analysis levels: %s", cfg["levels"])

        if cfg["tracks"]["semantic"].get("enabled", False):
            _run_track_analysis(
                track_name="semantic",
                task_names=cfg["tracks"]["semantic"]["tasks"],
                records=semantic_records,
                models=models,
                cfg=cfg,
                output_root=output_root,
                finetuned_by_family=None,
                probing_n_jobs=rt["probing_n_jobs"],
            )

        if cfg["tracks"]["tone"].get("enabled", False):
            _run_track_analysis(
                track_name="tone",
                task_names=cfg["tracks"]["tone"]["tasks"],
                records=tone_records,
                models=models,
                cfg=cfg,
                output_root=output_root,
                finetuned_by_family={m: tone_ckpts.get(m) for m in models},
                probing_n_jobs=rt["probing_n_jobs"],
            )

        if cfg["tracks"]["style"].get("enabled", False):
            _run_track_analysis(
                track_name="style",
                task_names=cfg["tracks"]["style"]["tasks"],
                records=style_records,
                models=models,
                cfg=cfg,
                output_root=output_root,
                finetuned_by_family={m: style_ckpts.get(m) for m in models},
                probing_n_jobs=rt["probing_n_jobs"],
            )

        log_section(log, "Done")
        log.info("All tasks finished. Log: %s", log_path.resolve())
    finally:
        crash_log.close()


if __name__ == "__main__":
    main()
