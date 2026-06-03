"""Извлечение hidden states"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.embeddings.normalization import NormalizationProtocol
from src.embeddings.pooling import apply_pooling
from src.embeddings.storage import EmbeddingStore
from src.language_models.backbone import TransformerBackbone

logger = logging.getLogger(__name__)


def set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HiddenStateExtractor:
    def __init__(
        self,
        backbone: TransformerBackbone,
        *,
        max_length: int = 256,
        pooling: str = "mean",
        normalization: NormalizationProtocol | None = None,
        use_amp: bool = False,
        seed: int = 42,
    ):
        self.backbone = backbone
        self.max_length = max_length
        self.pooling = pooling
        self.normalization = normalization or NormalizationProtocol(method="none")
        self.use_amp = use_amp
        set_deterministic(seed)

    def extract_batch(
        self,
        texts: list[str],
        *,
        output_attentions: bool = False,
        sample_offset: int = 0,
        sample_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        inputs = self.backbone.tokenize(texts, self.max_length)
        outputs = self.backbone.forward(
            inputs,
            output_hidden_states=True,
            output_attentions=output_attentions,
            use_amp=self.use_amp,
        )
        hidden = self.backbone.hidden_states(outputs)
        mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"].cpu().numpy()

        pooled_per_layer: list[np.ndarray] = []
        for layer_tensor in hidden:
            pooled = apply_pooling(layer_tensor, mask, self.pooling)
            pooled_per_layer.append(pooled.detach().cpu().numpy())

        attn_maps = None
        if output_attentions:
            att = self.backbone.attentions(outputs)
            if att:
                attn_maps = [a.detach().cpu().numpy() for a in att]

        meta_rows = []
        for i in range(len(texts)):
            sid = (sample_ids[i] if sample_ids else f"s{sample_offset + i}")
            meta_rows.append(
                {
                    "sample_id": sid,
                    "token_ids": input_ids[i].tolist(),
                    "attention_mask": mask[i].cpu().tolist(),
                    "text_length": len(texts[i]),
                }
            )

        return {
            "pooled_per_layer": pooled_per_layer,
            "hidden_states": [h.detach().cpu().numpy() for h in hidden],
            "attentions": attn_maps,
            "metadata": meta_rows,
        }

    def extract_and_store(
        self,
        texts: list[str],
        store: EmbeddingStore,
        *,
        batch_size: int = 8,
        chunk_size: int | None = None,
        cache_path: Path | None = None,
        sample_ids: list[str] | None = None,
        output_attentions: bool = False,
    ) -> EmbeddingStore:
        if cache_path and cache_path.exists():
            logger.info("Using cached extraction: %s", cache_path)
            return store

        n = len(texts)
        chunk = chunk_size or n
        all_pooled: list[list[np.ndarray]] | None = None

        for start in range(0, n, chunk):
            batch_texts = texts[start : start + chunk]
            batch_ids = sample_ids[start : start + chunk] if sample_ids else None
            for sub in range(0, len(batch_texts), batch_size):
                sub_texts = batch_texts[sub : sub + batch_size]
                sub_ids = batch_ids[sub : sub + batch_size] if batch_ids else None
                out = self.extract_batch(
                    sub_texts,
                    output_attentions=output_attentions,
                    sample_offset=start + sub,
                    sample_ids=sub_ids,
                )
                for meta in out["metadata"]:
                    store.append_sample_metadata(meta)

                if all_pooled is None:
                    n_layers = len(out["pooled_per_layer"])
                    all_pooled = [[] for _ in range(n_layers)]

                for li, mat in enumerate(out["pooled_per_layer"]):
                    all_pooled[li].append(mat)

        assert all_pooled is not None
        for li, parts in enumerate(all_pooled):
            matrix = np.concatenate(parts, axis=0)
            matrix = self.normalization.fit_transform(matrix)
            store.write_layer_pooled(li, matrix)

        norm_path = store.root / "normalization_protocol.json"
        self.normalization.save(norm_path)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"cached": True, "n": n}), encoding="utf-8")

        return store
