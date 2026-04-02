"""
Визуализация mean-pooled эмбеддингов по слоям (тональность / стиль — разные чекпоинты).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _truncate(text: str, max_len: int = 56) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _pca_path(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    color: str,
) -> None:
    """Траектория слоёв в 2D (PCA по одной траектории)."""
    n, d = matrix.shape
    if n < 2:
        ax.text(0.5, 0.5, "Нужно ≥2 слоёв", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    n_comp = min(2, n - 1, d)
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp)
    xy = pca.fit_transform(matrix.astype(np.float64))
    if xy.shape[1] >= 2:
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, alpha=0.85, linewidth=1.5)
        ax.scatter(xy[:, 0], xy[:, 1], c=range(n), cmap="viridis", s=36, zorder=3, edgecolors="white", linewidths=0.5)
        for i in range(n):
            ax.annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.9)
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    else:
        layers = np.arange(n, dtype=np.float64)
        ax.plot(xy[:, 0], layers, "-", color=color, alpha=0.85)
        ax.scatter(xy[:, 0], layers, c=range(n), cmap="viridis", s=36, zorder=3, edgecolors="white", linewidths=0.5)
        for i in range(n):
            ax.annotate(str(i), (xy[i, 0], layers[i]), fontsize=7, alpha=0.9)
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel("Индекс слоя")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _heatmap_layers_dims(ax: plt.Axes, matrix: np.ndarray, title: str, n_dims: int) -> None:
    """Теплокарта: слой × первые n_dims измерений (z-score по слоям для каждой размерности)."""
    n, d = matrix.shape
    k = min(n_dims, d)
    block = matrix[:, :k].astype(np.float64)
    if n >= 2:
        block = StandardScaler().fit_transform(block)
    else:
        block = block - block.mean()
    im = ax.imshow(block, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i) for i in range(n)])
    ax.set_xlabel("Индекс размерности (начало вектора)")
    ax.set_ylabel("Индекс слоя")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_tone_style_layer_embeddings(
    tone_layers: np.ndarray,
    style_layers: np.ndarray,
    *,
    text_preview: str,
    show: bool = True,
    save_path: Path | None = None,
    heatmap_dims: int = 48,
) -> None:
    """
    tone_layers, style_layers: форма (num_layers, hidden_dim) для каждого чекпоинта.
    Два ряда: PCA-траектория и теплокарта по слоям.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    suptitle = f"Эмбеддинги по слоям (mean pooling)\n{_truncate(text_preview)}"
    fig.suptitle(suptitle, fontsize=11)

    _pca_path(axes[0, 0], tone_layers, "Тональность: траектория слоёв в 2D (PCA)", "#1f77b4")
    _pca_path(axes[0, 1], style_layers, "Стиль (эмоции): траектория слоёв в 2D (PCA)", "#ff7f0e")

    _heatmap_layers_dims(axes[1, 0], tone_layers, f"Тональность: слой × dim (z-score), dim≤{heatmap_dims}", heatmap_dims)
    _heatmap_layers_dims(axes[1, 1], style_layers, f"Стиль: слой × dim (z-score), dim≤{heatmap_dims}", heatmap_dims)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
