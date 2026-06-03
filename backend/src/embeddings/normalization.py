"""Normalization subsystem: centering, L2, PCA removal, whitening, none."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class NormalizationProtocol:
    method: str = "none"
    pca_components: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    _pca: PCA | None = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)

    def fit(self, x: np.ndarray) -> NormalizationProtocol:
        x = np.asarray(x, dtype=np.float64)
        if self.method == "none":
            return self
        if self.method == "center":
            self._mean = x.mean(axis=0)
            return self
        if self.method in ("pca_remove", "whitening"):
            n_comp = self.pca_components or min(8, x.shape[1])
            self._pca = PCA(n_components=n_comp)
            self._pca.fit(x)
            return self
        if self.method == "l2":
            return self
        raise ValueError(f"Unknown normalization: {self.method}")

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.method == "none":
            return x
        if self.method == "center":
            assert self._mean is not None
            return x - self._mean
        if self.method == "l2":
            norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            return x / norms
        if self.method == "pca_remove" and self._pca is not None:
            proj = self._pca.transform(x)
            recon = self._pca.inverse_transform(proj)
            return x - recon
        if self.method == "whitening" and self._pca is not None:
            z = self._pca.transform(x)
            std = np.std(z, axis=0) + 1e-12
            return z / std
        return x

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "pca_components": self.pca_components,
            "extra": self.extra,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | str | None) -> NormalizationProtocol:
        if cfg is None or cfg == "none":
            return cls(method="none")
        if isinstance(cfg, str):
            return cls(method=cfg)
        return cls(
            method=str(cfg.get("method", "none")),
            pca_components=cfg.get("pca_components"),
            extra=dict(cfg.get("extra") or {}),
        )
