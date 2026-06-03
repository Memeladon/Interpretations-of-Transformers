"""Единый интерфейс encoder/decoder трансформеров"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.language_models.loader import MODEL_REGISTRY

logger = logging.getLogger(__name__)

ArchitectureKind = Literal["encoder", "decoder"]

DECODER_FAMILIES = frozenset({"gpt", "llama", "gpt2", "decoder"})


class TransformerBackbone:
    """Унифицированный доступ: hidden states, attentions, token embeddings"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        *,
        family: str,
        model_id: str,
        architecture: ArchitectureKind,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.family = family
        self.model_id = model_id
        self.architecture = architecture
        self.model.eval()

    @classmethod
    def from_family(
        cls,
        family: str,
        *,
        model_name: str | None = None,
        device: str | None = None,
        finetuned_checkpoint: str | Path | None = None,
        attn_implementation: str | None = None,
    ) -> TransformerBackbone:
        family_l = family.lower()
        model_id = model_name or MODEL_REGISTRY.get(family_l, family_l)
        arch: ArchitectureKind = "decoder" if family_l in DECODER_FAMILIES else "encoder"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        kw = {}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation

        if finetuned_checkpoint is not None:
            ckpt = Path(finetuned_checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForSequenceClassification.from_pretrained(str(ckpt), **kw)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(model_id, **kw)

        model.to(device)
        model.eval()
        return cls(model, tokenizer, device, family=family_l, model_id=str(model_id), architecture=arch)

    def tokenize(self, texts: list[str], max_length: int, padding: str = "max_length"):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        use_amp: bool = False,
    ) -> Any:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            if use_amp and self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(
                        **inputs,
                        output_hidden_states=output_hidden_states,
                        output_attentions=output_attentions,
                    )
            return self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

    @staticmethod
    def hidden_states(outputs: Any) -> tuple[torch.Tensor, ...]:
        hs = getattr(outputs, "hidden_states", None)
        if hs is None:
            raise ValueError("Model output has no hidden_states")
        return tuple(hs)

    @staticmethod
    def attentions(outputs: Any) -> tuple[torch.Tensor, ...] | None:
        return getattr(outputs, "attentions", None)

    @staticmethod
    def token_embeddings(outputs: Any, layer: int = 0) -> torch.Tensor:
        return TransformerBackbone.hidden_states(outputs)[layer]
