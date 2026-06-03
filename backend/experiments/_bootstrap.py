"""Общая инициализация entrypoint-скриптов"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> Path:
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    if os.name == "nt":
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(key, "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    return BACKEND_ROOT


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    from src.paths import ProjectPaths

    paths = ProjectPaths.from_root(BACKEND_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=paths.default_config_path(),
        help="YAML/JSON конфиг (по умолчанию configs/experiment.yaml)",
    )


def add_run_id_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", type=str, default=None, help="Идентификатор запуска (runs/<id>/)")
