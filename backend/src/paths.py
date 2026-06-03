"""Корневые пути проекта"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    configs: Path
    data: Path
    data_raw: Path
    data_processed: Path
    data_splits: Path
    artifacts: Path
    src: Path
    reports: Path
    ui: Path
    logs: Path
    runs: Path
    experiments: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        base = (root or BACKEND_ROOT).resolve()
        data = base / "data"
        return cls(
            root=base,
            configs=base / "configs",
            data=data,
            data_raw=data / "raw",
            data_processed=data / "processed",
            data_splits=data / "splits",
            artifacts=base / "artifacts",
            src=base / "src",
            reports=base / "reports",
            ui=base / "ui",
            logs=base / "logs",
            runs=base / "runs",
            experiments=base / "experiments",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.configs,
            self.data_raw,
            self.data_processed,
            self.data_splits,
            self.artifacts,
            self.reports,
            self.ui,
            self.logs,
            self.runs,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def default_config_path(self) -> Path:
        yaml_path = self.configs / "experiment.yaml"
        if yaml_path.exists():
            return yaml_path
        legacy = self.experiments / "experiment_config.json"
        if legacy.exists():
            return legacy
        return yaml_path
