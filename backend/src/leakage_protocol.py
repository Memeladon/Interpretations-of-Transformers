"""Leakage protocol: автоматический контроль утечек между train/val/test"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.data_schema import normalize_text_key

logger = logging.getLogger(__name__)


def leakage_group_key(row: dict[str, Any]) -> str:
    """
    Ключ группы для split без утечки: одна pair → один pair_id
    """
    meta = row.get("metadata") or {}
    pair_id = meta.get("pair_id")
    if pair_id:
        return f"pair_id:{pair_id}"

    parts: list[str] = []
    text = normalize_text_key(str(row.get("text", "")))
    if text:
        parts.append(text)
    text_pair = row.get("text_pair")
    if text_pair:
        parts.append(normalize_text_key(str(text_pair)))
    if parts:
        return "text:" + "||".join(sorted(parts))
    return f"row:{row.get('id', meta.get('record_id', 'unknown'))}"


def _row_text_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    text = normalize_text_key(str(row.get("text", "")))
    if text:
        keys.add(text)
    text_pair = row.get("text_pair")
    if text_pair:
        keys.add(normalize_text_key(str(text_pair)))
    return keys


def group_indices_by_leakage_key(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    """
    Группы связных компонент: строки объединяются, если делят pair_id
    или хотя бы один нормализованный text/text_pair.
    """
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    text_anchor: dict[str, int] = {}
    pair_anchor: dict[str, int] = {}

    for i, row in enumerate(rows):
        meta = row.get("metadata") or {}
        pair_id = meta.get("pair_id")
        if pair_id:
            pid = str(pair_id)
            if pid in pair_anchor:
                union(i, pair_anchor[pid])
            else:
                pair_anchor[pid] = i

        for key in _row_text_keys(row):
            if key in text_anchor:
                union(i, text_anchor[key])
            else:
                text_anchor[key] = i

    components: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, []).append(i)

    return {f"component:{root}": members for root, members in components.items()}


@dataclass
class LeakageReport:
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "violations": self.violations, "warnings": self.warnings}


def _partition_texts(rows: list[dict[str, Any]], indices: dict[str, list[int]]) -> dict[str, set[str]]:
    parts: dict[str, set[str]] = {}
    for part, idxs in indices.items():
        texts = set()
        for i in idxs:
            if 0 <= i < len(rows):
                row = rows[i]
                texts.add(normalize_text_key(str(row.get("text", ""))))
                text_pair = row.get("text_pair")
                if text_pair:
                    texts.add(normalize_text_key(str(text_pair)))
        parts[part] = texts
    return parts


def _partition_pair_ids(rows: list[dict[str, Any]], indices: dict[str, list[int]]) -> dict[str, set[str]]:
    parts: dict[str, set[str]] = {}
    for part, idxs in indices.items():
        ids: set[str] = set()
        for i in idxs:
            if 0 <= i < len(rows):
                meta = rows[i].get("metadata") or {}
                pid = meta.get("pair_id")
                if pid:
                    ids.add(str(pid))
        parts[part] = ids
    return parts


def _partition_sources(rows: list[dict[str, Any]], indices: dict[str, list[int]]) -> dict[str, set[str]]:
    parts: dict[str, set[str]] = {}
    for part, idxs in indices.items():
        keys: set[str] = set()
        for i in idxs:
            if 0 <= i < len(rows):
                meta = rows[i].get("metadata") or {}
                rid = meta.get("record_id") or rows[i].get("id")
                src = meta.get("source", "")
                keys.add(f"{src}::{rid}")
        parts[part] = keys
    return parts


def check_split_leakage(
    rows: list[dict[str, Any]],
    indices: dict[str, list[int]],
    *,
    check_duplicate_text: bool = True,
    check_pair_id_overlap: bool = True,
    check_source_overlap: bool = True,
) -> LeakageReport:
    report = LeakageReport()
    parts = list(indices.keys())

    if check_duplicate_text:
        texts_by_part = _partition_texts(rows, indices)
        for i, a in enumerate(parts):
            for b in parts[i + 1 :]:
                overlap = texts_by_part[a] & texts_by_part[b]
                overlap.discard("")
                if overlap:
                    report.violations.append(
                        f"duplicate text between {a} and {b}: {len(overlap)} example(s)"
                    )

    if check_pair_id_overlap:
        pairs_by_part = _partition_pair_ids(rows, indices)
        for i, a in enumerate(parts):
            for b in parts[i + 1 :]:
                overlap = pairs_by_part[a] & pairs_by_part[b]
                if overlap:
                    report.violations.append(
                        f"pair_id overlap between {a} and {b}: {len(overlap)} id(s)"
                    )

    if check_source_overlap:
        src_by_part = _partition_sources(rows, indices)
        for i, a in enumerate(parts):
            for b in parts[i + 1 :]:
                overlap = src_by_part[a] & src_by_part[b]
                if overlap:
                    report.warnings.append(
                        f"source+record_id overlap between {a} and {b}: {len(overlap)} key(s)"
                    )

    return report


def enforce_leakage_protocol(
    rows: list[dict[str, Any]],
    indices: dict[str, list[int]],
    leakage_cfg: dict[str, Any],
) -> LeakageReport:
    if not leakage_cfg.get("enabled", True):
        return LeakageReport()

    report = check_split_leakage(
        rows,
        indices,
        check_duplicate_text=bool(leakage_cfg.get("check_duplicate_text", True)),
        check_pair_id_overlap=bool(leakage_cfg.get("check_pair_id_overlap", True)),
        check_source_overlap=bool(leakage_cfg.get("check_source_overlap", True)),
    )
    for w in report.warnings:
        logger.warning("leakage warning: %s", w)
    for v in report.violations:
        logger.error("leakage violation: %s", v)
    if report.violations and leakage_cfg.get("fail_on_violation", True):
        raise RuntimeError(
            "Leakage protocol failed: " + "; ".join(report.violations)
        )
    return report
