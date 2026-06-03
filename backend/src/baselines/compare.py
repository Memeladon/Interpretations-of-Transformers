"""Сравнение probing с leakage baselines и предупреждения"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compare_probing_to_baselines(
    probing_scores: dict[str, float],
    baseline_results: dict[str, Any],
    *,
    tolerance: float = 0.05,
    metric_name: str = "primary_score",
) -> dict[str, Any]:
    """
    Если baseline ≈ probing (в пределах tolerance), формирует предупреждение
    о возможной утечке через поверхностные признаки.
    """
    if not probing_scores:
        return {"warnings": [], "ok": True}

    best_probe_layer = max(probing_scores, key=lambda k: probing_scores[k])
    best_probe_score = float(probing_scores[best_probe_layer])

    warnings: list[str] = []
    comparisons: list[dict[str, Any]] = []

    for est_name, est_data in (baseline_results.get("estimators") or {}).items():
        baseline_score = float(est_data.get("primary_score", est_data.get("metrics", {}).get(metric_name, 0)))
        gap = best_probe_score - baseline_score
        comparable = baseline_score >= best_probe_score * (1.0 - tolerance) or abs(gap) <= tolerance
        entry = {
            "estimator": est_name,
            "baseline_score": baseline_score,
            "best_probe_layer": best_probe_layer,
            "best_probe_score": best_probe_score,
            "gap": gap,
            "comparable": comparable,
        }
        comparisons.append(entry)
        if comparable:
            msg = (
                f"LEAKAGE WARNING: surface baseline '{est_name}' ({baseline_score:.4f}) is "
                f"comparable to best probing layer {best_probe_layer} ({best_probe_score:.4f}); "
                f"representation signal may not exceed superficial features."
            )
            warnings.append(msg)
            logger.warning(msg)

    report = {
        "ok": len(warnings) == 0,
        "warnings": warnings,
        "comparisons": comparisons,
        "best_probe_layer": best_probe_layer,
        "best_probe_score": best_probe_score,
        "tolerance": tolerance,
    }
    return report


def save_comparison_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
