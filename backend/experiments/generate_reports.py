"""Генерация отчётов и визуализаций"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _bootstrap import add_config_arg, bootstrap

bootstrap()

from src.experiment_config import load_experiment_config
from src.experiment_logging import log_section, setup_experiment_logging
from src.paths import ProjectPaths
from src.reports import generate_all_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация отчётов из artifacts/analysis")
    add_config_arg(parser)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Каталог analysis (по умолчанию artifacts/analysis)",
    )
    parser.add_argument("--robustness", type=Path, default=None, help="JSON устойчивости")
    args = parser.parse_args()

    paths = ProjectPaths.from_root()
    cfg = load_experiment_config(args.config)
    setup_experiment_logging(paths.logs)
    log = logging.getLogger("generate_reports")
    analysis_dir = args.analysis_dir or (paths.artifacts / "analysis")
    robustness = args.robustness or (paths.artifacts / "robustness.json")

    log_section(log, "Stage: generate reports")
    out = generate_all_reports(
        analysis_dir,
        paths.reports,
        robustness_path=robustness if robustness.exists() else None,
    )
    for name, path in out.items():
        log.info("%s → %s", name, path.resolve())


if __name__ == "__main__":
    main()
