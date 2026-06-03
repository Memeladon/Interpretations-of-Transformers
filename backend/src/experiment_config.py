from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Для YAML-конфигов установите PyYAML: uv add pyyaml") from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def load_experiment_config(path: str | Path, *, resolve: bool = True) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = _load_raw(config_path)
    if not resolve:
        return raw
    from src.config_resolve import resolve_experiment_config

    return resolve_experiment_config(raw)
