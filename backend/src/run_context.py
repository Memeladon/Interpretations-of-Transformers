"""Контекст одного воспроизводимого запуска: runs/<run_id>/"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paths import ProjectPaths


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    paths: ProjectPaths
    config_path: Path
    config: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        config: dict[str, Any],
        config_path: Path,
        paths: ProjectPaths | None = None,
        run_id: str | None = None,
    ) -> RunContext:
        project = paths or ProjectPaths.from_root()
        project.ensure_dirs()
        rid = run_id or config.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = project.runs / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, config_dir / config_path.name)
        manifest = {
            "run_id": rid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_source": str(config_path.resolve()),
            "seed": config.get("seed"),
            "stages_completed": [],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cls(run_id=rid, run_dir=run_dir, paths=project, config_path=config_path, config=config)

    def log_dir(self) -> Path:
        return self.paths.logs

    def artifact_root(self) -> Path:
        """Корень артефактов для этого запуска (подкаталог в artifacts/)."""
        root = self.paths.artifacts / "runs" / self.run_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def mark_stage(self, stage: str) -> None:
        manifest_path = self.run_dir / "manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages: list[str] = data.setdefault("stages_completed", [])
        if stage not in stages:
            stages.append(stage)
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
