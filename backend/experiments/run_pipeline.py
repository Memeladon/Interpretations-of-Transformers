"""Полный исследовательский pipeline"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os

import numpy as np
import torch

from _bootstrap import add_config_arg, add_run_id_arg, bootstrap

bootstrap()

from src.data_splits import build_all_splits  # noqa: E402
from src.data_pipeline import ensure_processed_tracks  # noqa: E402
from src.experiment_config import load_experiment_config  # noqa: E402
from src.experiment_logging import log_section, setup_experiment_logging  # noqa: E402
from src.experiment_runner import resolve_runtime, run_full_analysis  # noqa: E402
from src.paths import ProjectPaths  # noqa: E402
from src.reports import generate_all_reports  # noqa: E402
from src.robustness import robustness_from_analysis_summaries  # noqa: E402
from src.run_context import RunContext  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный pipeline интерпретации")
    add_config_arg(parser)
    add_run_id_arg(parser)
    parser.add_argument("--skip-prepare-data", action="store_true", help="Не пересобирать data/processed/")
    parser.add_argument(
        "--force-prepare-data",
        action="store_true",
        help="Пересобрать data/processed/ даже если файлы уже есть",
    )
    parser.add_argument("--skip-splits", action="store_true", help="Не пересобирать splits")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_root()
    paths.ensure_dirs()
    cfg = load_experiment_config(args.config)
    run = RunContext.create(config=cfg, config_path=args.config, paths=paths, run_id=args.run_id)
    log_path = setup_experiment_logging(paths.logs)
    log = logging.getLogger("run_pipeline")
    crash_log = open(log_path, "a", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)

    try:
        log_section(log, "Transformer interpretation — full pipeline")
        log.info("Config: %s | Run: %s", args.config.resolve(), run.run_id)
        set_seed(int(cfg["seed"]))
        rt = resolve_runtime(cfg)
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if rt["tokenizers_parallelism"] else "false"
        torch.set_num_threads(rt["cpu_threads"])
        torch.set_num_interop_threads(rt["torch_interop_threads"])

        if not args.skip_prepare_data:
            log_section(log, "Stage: prepare data")
            ensure_processed_tracks(cfg, paths=paths, force=args.force_prepare_data)
            run.mark_stage("prepare_data")
        else:
            log.info("Skipping prepare data (--skip-prepare-data)")

        if not args.skip_splits:
            log_section(log, "Stage: build splits")
            build_all_splits(cfg, paths=paths)
            run.mark_stage("build_splits")

        artifact_root = run.artifact_root()
        log_section(log, "Stage: embeddings + probing + interventions")
        run_full_analysis(cfg, paths=paths, artifact_root=artifact_root, use_processed_data=True)
        run.mark_stage("analysis")

        analysis_dir = artifact_root / "analysis"
        robustness_path = artifact_root / "robustness.json"
        rob_cfg = cfg.get("robustness") or {}
        if not args.skip_robustness and rob_cfg.get("enabled", True) and analysis_dir.exists():
            log_section(log, "Stage: robustness")
            results = robustness_from_analysis_summaries(
                analysis_dir,
                bootstrap_seeds=[int(s) for s in rob_cfg.get("bootstrap_seeds", [42, 43, 44])],
                subsample_fraction=float(rob_cfg.get("subsample_fraction", 0.8)),
                probing_n_jobs=rt["probing_n_jobs"],
            )
            robustness_path.write_text(
                __import__("json").dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for key, payload in results.items():
                layers = payload.get("layers") or {}
                if not layers:
                    continue
                best_layer = max(layers.keys(), key=lambda k: layers[k].get("mean", 0))
                stats = layers[best_layer]
                log.info(
                    "Robustness %s layer=%s: mean=%.4f std=%.4f ci95=[%.4f, %.4f] (seeds=%s)",
                    key,
                    best_layer,
                    stats["mean"],
                    stats["std"],
                    stats.get("ci95_low", stats["min"]),
                    stats.get("ci95_high", stats["max"]),
                    stats.get("n_seeds", len(rob_cfg.get("bootstrap_seeds", []))),
                )
            run.mark_stage("robustness")

        if not args.skip_reports:
            log_section(log, "Stage: reports")
            generate_all_reports(
                analysis_dir,
                paths.reports,
                robustness_path=robustness_path if robustness_path.exists() else None,
            )
            run.mark_stage("reports")

        log_section(log, "Done")
        log.info("Artifacts: %s | Reports: %s | Log: %s", artifact_root, paths.reports, log_path)
    finally:
        crash_log.close()


if __name__ == "__main__":
    main()
