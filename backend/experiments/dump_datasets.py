"""
Дамп записей из текущих датасетов (semantic/tone/style) в backend/data/raw/.
По умолчанию используется та же схема и маппинг, что и pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_config_arg, add_run_id_arg, bootstrap

bootstrap()

from src.data_schema import normalize_text_key
from src.datasets_load import load_track_datasets
from src.experiment_config import load_experiment_config
from src.experiment_logging import log_section, setup_experiment_logging
from src.paths import ProjectPaths
from src.run_context import RunContext

log = logging.getLogger("dump_datasets")


def _enabled_tracks_from_cfg(cfg: dict) -> list[str]:
    tracks_cfg = cfg.get("tracks") or {}
    out: list[str] = []
    for name in ("semantic", "tone", "style"):
        if tracks_cfg.get(name, {}).get("enabled", False):
            out.append(name)
    return out


def _dump_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump текущих датасетов в data/raw/.")
    add_config_arg(parser)
    add_run_id_arg(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Куда писать дамп (по умолчанию: data/raw/dataset_dump/<timestamp>).",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Лимит строк на исходный HF датасет после маппинга (None → взять dataset_limit_per_source из конфига).",
    )
    args = parser.parse_args()

    paths = ProjectPaths.from_root()
    cfg = load_experiment_config(args.config, resolve=True)  # type: ignore[arg-type]
    run = RunContext.create(config=cfg, config_path=args.config, paths=paths, run_id=args.run_id)

    log_dir = paths.logs
    setup_experiment_logging(log_dir)

    log_section(log, "Dump datasets")
    enabled = _enabled_tracks_from_cfg(cfg)
    if not enabled:
        log.error("No enabled tracks found in config: %s", args.config)
        raise SystemExit(1)

    limit_per_dataset = args.limit_per_dataset
    if limit_per_dataset is None:
        limit_per_dataset = int(cfg.get("dataset_limit_per_source", 600))
    seed = int(cfg.get("seed", 42))
    cache_dir = cfg.get("cache_dir", ".cache/hf_datasets")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_out = args.output_dir or (paths.data_raw / "dataset_dump" / (run.run_id or ts))
    base_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run.run_id,
        "config_path": str(Path(args.config).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "enabled_tracks": enabled,
        "seed": seed,
        "cache_dir": cache_dir,
        "limit_per_dataset": limit_per_dataset,
        "files": {},
    }

    for track in enabled:
        rows = load_track_datasets(
            track,
            cache_dir=cache_dir,
            limit_per_dataset=limit_per_dataset,
            seed=seed,
        )
        # Мини-лайфхак для удобства проверки: нормализованный текст-ключ (не TЗ, просто для отладки).
        for r in rows:
            r["_text_key"] = normalize_text_key(r.get("text", "") or "")
        fname = f"{track}_records.jsonl"
        fpath = base_out / fname
        n = _dump_jsonl(rows, fpath)
        manifest["files"][track] = {"path": str(fpath), "n": n}
        log.info("Dumped %s: n=%s → %s", track, n, fpath)

    (base_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Done. Manifest: %s", (base_out / "manifest.json").resolve())


if __name__ == "__main__":
    main()

