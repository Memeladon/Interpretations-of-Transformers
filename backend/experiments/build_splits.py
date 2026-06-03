"""Этап 2: формирование train/val/test splits → data/splits/"""

from __future__ import annotations

import argparse
import logging

from _bootstrap import add_config_arg, add_run_id_arg, bootstrap

bootstrap()

from src.data_splits import build_all_splits
from src.experiment_config import load_experiment_config
from src.experiment_logging import log_section, setup_experiment_logging
from src.paths import ProjectPaths
from src.run_context import RunContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Формирование splits")
    add_config_arg(parser)
    add_run_id_arg(parser)
    args = parser.parse_args()

    paths = ProjectPaths.from_root()
    cfg = load_experiment_config(args.config)
    run = RunContext.create(config=cfg, config_path=args.config, paths=paths, run_id=args.run_id)
    log_path = setup_experiment_logging(paths.logs)
    log = logging.getLogger("build_splits")
    log_section(log, "Stage: build splits")
    log.info("Run: %s | Log: %s", run.run_id, log_path)

    build_all_splits(cfg, paths=paths)
    run.mark_stage("build_splits")
    log.info("Done. Splits in %s", paths.data_splits.resolve())


if __name__ == "__main__":
    main()
