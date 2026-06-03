"""Этап 1: загрузка и предобработка датасетов → data/processed/"""

from __future__ import annotations

import argparse
import logging

from _bootstrap import add_config_arg, add_run_id_arg, bootstrap

bootstrap()

from src.data_pipeline import ensure_processed_tracks
from src.experiment_config import load_experiment_config
from src.experiment_logging import log_section, setup_experiment_logging
from src.paths import ProjectPaths
from src.run_context import RunContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка данных (HF → data/processed/)")
    add_config_arg(parser)
    add_run_id_arg(parser)
    args = parser.parse_args()

    paths = ProjectPaths.from_root()
    paths.ensure_dirs()
    cfg = load_experiment_config(args.config)
    run = RunContext.create(config=cfg, config_path=args.config, paths=paths, run_id=args.run_id)
    log_path = setup_experiment_logging(paths.logs)
    log = logging.getLogger("prepare_data")
    log_section(log, "Stage: prepare data")
    log.info("Config: %s | Run: %s | Log: %s", args.config, run.run_id, log_path)

    ensure_processed_tracks(cfg, paths=paths, force=True)
    run.mark_stage("prepare_data")
    log.info("Done. Processed files in %s", paths.data_processed.resolve())


if __name__ == "__main__":
    main()
