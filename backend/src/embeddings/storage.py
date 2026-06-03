"""Хранилище эмбеддингов (тензоры отдельно)"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRun:
    run_id: str
    model_id: str
    dataset_id: str
    layer_list: list[int]
    pooling_type: str
    normalization_type: str
    extraction_timestamp: str
    config_hash: str
    n_samples: int = 0
    hidden_dim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingStore:
    """
    Структура:
      <root>/manifest.json
      meta.jsonl
      layers/layer_XX/pooled.npy   # memmap (N, D)
      layers/layer_XX/attn/        # optional
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest: EmbeddingRun | None = None
        self._mmap_cache: dict[str, np.memmap] = {}

    @staticmethod
    def config_hash(cfg: dict[str, Any]) -> str:
        blob = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def init_run(self, run: EmbeddingRun) -> None:
        self._manifest = run
        (self.root / "manifest.json").write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.root / "meta.jsonl").write_text("", encoding="utf-8")

    def append_sample_metadata(self, record: dict[str, Any]) -> None:
        with (self.root / "meta.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_layer_pooled(
        self,
        layer_idx: int,
        matrix: np.ndarray,
        *,
        mode: str = "w",
    ) -> Path:
        layer_dir = self.root / "layers" / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir / "pooled.npy"
        if mode == "w":
            np.save(path, matrix)
        else:
            # append via memmap resize not implemented — overwrite per run
            np.save(path, matrix)
        return path

    def open_layer_mmap(self, layer_idx: int, mode: str = "r") -> np.memmap:
        key = f"layer_{layer_idx:02d}"
        if key in self._mmap_cache:
            return self._mmap_cache[key]
        path = self.root / "layers" / key / "pooled.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path, mmap_mode=mode)
        self._mmap_cache[key] = arr
        return arr

    def get_batch(self, layer_idx: int, indices: list[int]) -> np.ndarray:
        mmap = self.open_layer_mmap(layer_idx)
        return np.asarray(mmap[indices])

    def layer_indices_available(self) -> list[int]:
        layers_dir = self.root / "layers"
        if not layers_dir.exists():
            return []
        out = []
        for p in sorted(layers_dir.glob("layer_*")):
            try:
                out.append(int(p.name.split("_")[1]))
            except ValueError:
                continue
        return out

    @classmethod
    def create_run(
        cls,
        root: Path,
        *,
        model_id: str,
        dataset_id: str,
        layer_list: list[int],
        pooling_type: str,
        normalization_type: str,
        config: dict[str, Any],
        n_samples: int = 0,
        hidden_dim: int = 0,
        run_id: str | None = None,
    ) -> EmbeddingStore:
        rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run = EmbeddingRun(
            run_id=rid,
            model_id=model_id,
            dataset_id=dataset_id,
            layer_list=layer_list,
            pooling_type=pooling_type,
            normalization_type=normalization_type,
            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=cls.config_hash(config),
            n_samples=n_samples,
            hidden_dim=hidden_dim,
        )
        store = cls(root / rid)
        store.init_run(run)
        return store
