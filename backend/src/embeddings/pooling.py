"""Pooling subsystem: CLS, mean, max, attention, custom."""

from __future__ import annotations

from typing import Callable, Protocol

import torch


class PoolingFn(Protocol):
    def __call__(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def cls_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


def max_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(hidden.size())
    masked = hidden.masked_fill(~m.bool(), float("-inf"))
    return masked.max(dim=1).values


def attention_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Self-attention weights = softmax over valid token dot-products with mean query."""
    m = mask.float()
    query = (hidden * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1e-9)
    scores = torch.bmm(hidden, query.unsqueeze(-1)).squeeze(-1)
    scores = scores.masked_fill(~mask.bool(), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)


POOLING_REGISTRY: dict[str, PoolingFn] = {
    "mean": mean_pooling,
    "cls": cls_pooling,
    "max": max_pooling,
    "attention": attention_pooling,
}


def get_pooling(name: str) -> PoolingFn:
    if name not in POOLING_REGISTRY:
        raise ValueError(f"Unknown pooling: {name}. Choose from {list(POOLING_REGISTRY)}")
    return POOLING_REGISTRY[name]


def apply_pooling(hidden: torch.Tensor, mask: torch.Tensor, strategy: str) -> torch.Tensor:
    return get_pooling(strategy)(hidden, mask)
