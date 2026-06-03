"""Генерация отчётов и визуализаций"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _collect_summaries(analysis_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(analysis_root.rglob("pipeline_summary.json")):
        rel = summary_path.parent.relative_to(analysis_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        probing = data.get("probing") or {}
        best_layer, best_score = None, None
        if probing:
            best_layer = max(probing, key=lambda k: probing[k])
            best_score = float(probing[best_layer])
        rows.append(
            {
                "track": parts[0],
                "task": parts[1],
                "family": parts[2],
                "level": parts[3],
                "best_layer": best_layer,
                "best_score": best_score,
                "probing": {str(k): float(v) for k, v in probing.items()},
                "intervention": data.get("intervention"),
                "decomposition": data.get("decomposition"),
                "summary_path": str(summary_path),
            }
        )
    return rows


def write_layer_profile_report(analysis_root: Path, reports_dir: Path) -> Path:
    """Layer-wise профили информативности → reports/layer_profiles.json."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_summaries(analysis_root)
    dest = reports_dir / "layer_profiles.json"
    dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Layer profiles report: %s (%s entries)", dest.resolve(), len(rows))
    return dest


def write_intervention_report(analysis_root: Path, reports_dir: Path) -> Path:
    dest = reports_dir / "intervention_summary.json"
    rows = []
    for summary_path in sorted(analysis_root.rglob("pipeline_summary.json")):
        rel = summary_path.parent.relative_to(analysis_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if not data.get("intervention"):
            continue
        rows.append(
            {
                "track": parts[0],
                "task": parts[1],
                "family": parts[2],
                "level": parts[3],
                "intervention": data["intervention"],
            }
        )
    dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Intervention report: %s", dest.resolve())
    return dest


def write_concept_directions_index(analysis_root: Path, reports_dir: Path) -> Path:
    dest = reports_dir / "concept_directions_index.json"
    rows = []
    for concept_path in sorted(analysis_root.rglob("concept_directions.json")):
        rel = concept_path.parent.relative_to(analysis_root)
        meta = json.loads(concept_path.read_text(encoding="utf-8"))
        rows.append({"path": str(concept_path), "relative": str(rel), **meta})
    dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Concept directions index: %s (%s)", dest.resolve(), len(rows))
    return dest


def generate_all_reports(
    analysis_root: Path,
    reports_dir: Path,
    *,
    robustness_path: Path | None = None,
) -> dict[str, Path]:
    out = {
        "layer_profiles": write_layer_profile_report(analysis_root, reports_dir),
        "interventions": write_intervention_report(analysis_root, reports_dir),
        "concept_directions": write_concept_directions_index(analysis_root, reports_dir),
    }
    if robustness_path and robustness_path.exists():
        data = json.loads(robustness_path.read_text(encoding="utf-8"))
        dest = reports_dir / "robustness_summary.json"
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        out["robustness"] = dest
    return out
