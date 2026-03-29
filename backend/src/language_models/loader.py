from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, str] = {
    "bert": "bert-base-multilingual-cased",
    "gpt": "ai-forever/rugpt3small_based_on_gpt2",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def load_language_model(
    model_family: str,
    model_name: str | None = None,
    device: str | None = None,
    finetuned_checkpoint: str | Path | None = None,
) -> tuple[Any, Any, str]:
    """
    Загрузка модели для извлечения hidden states.
    - Без чекпоинта: AutoModel (энкодер / базовый backbone).
    - С чекпоинтом: AutoModelForSequenceClassification (после fine-tuning);
      forward с output_hidden_states=True поддерживается.
    """
    family = model_family.lower()
    if model_name is None:
        if family not in MODEL_REGISTRY:
            known = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model family '{model_family}'. Use one of: {known}")
        model_name = MODEL_REGISTRY[family]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if finetuned_checkpoint is not None:
        ckpt = Path(finetuned_checkpoint)
        logger.info("loading finetuned classifier tokenizer: %s", ckpt.resolve())
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("loading finetuned classifier weights: %s -> %s", ckpt.resolve(), device)
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        model.eval()
        model.to(device)
        logger.info("model ready (finetuned): family=%s path=%s", family, ckpt.resolve())
        return model, tokenizer, device

    logger.info("loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading weights: %s -> %s", model_name, device)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    logger.info("model ready: family=%s name=%s", family, model_name)
    return model, tokenizer, device


def base_model_name_for_family(model_family: str) -> str:
    family = model_family.lower()
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: {model_family}")
    return MODEL_REGISTRY[family]
