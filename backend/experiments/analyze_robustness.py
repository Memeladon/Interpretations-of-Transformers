"""Этап 4: Анализ устойчивости layer-wise probing"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _bootstrap import add_config_arg, bootstrap

bootstrap()

from src.experiment_config import load_experiment_config
from src.experiment_logging import log_section, setup_experiment_logging
from src.experiment_runner import resolve_runtime
from src.paths import ProjectPaths
from src.robustness import robustness_from_analysis_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness analysis")
    add_config_arg(parser)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    paths = ProjectPaths.from_root()
    cfg = load_experiment_config(args.config)
    rt = resolve_runtime(cfg)
    setup_experiment_logging(paths.logs)
    log = logging.getLogger("analyze_robustness")

    analysis_dir = args.analysis_dir or (paths.artifacts / "analysis")
    rob_cfg = cfg.get("robustness") or {}
    log_section(log, "Stage: robustness")
    results = robustness_from_analysis_summaries(
        analysis_dir,
        bootstrap_seeds=[int(s) for s in rob_cfg.get("bootstrap_seeds", [42, 43, 44])],
        subsample_fraction=float(rob_cfg.get("subsample_fraction", 0.8)),
        probing_n_jobs=rt["probing_n_jobs"],
    )
    dest = args.output or (paths.artifacts / "robustness.json")
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Saved %s (%s keys)", dest.resolve(), len(results))


if __name__ == "__main__":
    main()
