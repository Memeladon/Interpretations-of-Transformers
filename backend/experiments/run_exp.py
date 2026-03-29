from __future__ import annotations

import logging
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

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


def _run_track_analysis(
    *,
    track_name: str,
    task_names: list[str],
    records: list[dict],
    models: list[str],
    cfg: dict,
    output_root: Path,
    finetuned_by_family: dict[str, Path | None] | None = None,
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
                log.info("Analysis level: %s", level)
                result = run_embedding_pipeline(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    labels=labels,
                    output_dir=level_out_dir,
                    probing_task_type=task_type,
                    level=level,
                    strategy=cfg["text_strategy"],
                    batch_size=cfg["batch_size"],
                    max_length=cfg["max_length"],
                    seed=cfg["seed"],
                    decomposition_methods=cfg["decomposition"]["enabled_methods"],
                    intervention_methods=cfg["decomposition"]["interventions"],
                    pca_components=cfg["decomposition"]["pca_components"],
                    drop_components=cfg["decomposition"]["drop_components"],
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

    set_seed(cfg["seed"])
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
        )

    log_section(log, "Done")
    log.info("All tasks finished. Log: %s", log_path.resolve())


if __name__ == "__main__":
    main()
