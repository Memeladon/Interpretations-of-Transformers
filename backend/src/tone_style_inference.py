"""
Инференс для дообученных голов тональности (single-label) и эмоций (multi-label),
совместимых с `finetune_classifier` в `finetuning.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.embeddings.aggregation import mean_pooling

# Соответствует полю `sentiment` в MonoHime/ru_sentiment_dataset (см. карточку датасета на HF).
RU_SENTIMENT_ID_TO_RU: dict[int, str] = {
    0: "нейтральная",
    1: "позитивная",
    2: "негативная",
}


def checkpoint_dir_ready(ckpt: Path) -> bool:
    return (ckpt / "config.json").exists() and (
        (ckpt / "pytorch_model.bin").exists() or (ckpt / "model.safetensors").exists()
    )


def load_classification_head(ckpt: Path, device: torch.device) -> tuple[Any, Any]:
    ckpt = Path(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.to(device)
    model.eval()
    return tokenizer, model


def _id2label_map(config: Any) -> dict[int, str]:
    raw = getattr(config, "id2label", None) or {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        out[idx] = str(v)
    return out


def sentiment_display_name(pred_id: int, config: Any) -> str:
    cfg_map = _id2label_map(config)
    name = cfg_map.get(pred_id)
    if name and not name.startswith("LABEL_"):
        return name
    return RU_SENTIMENT_ID_TO_RU.get(pred_id, f"класс_{pred_id}")


def sentiment_from_logits(logits: torch.Tensor, config: Any) -> tuple[str, int, float]:
    probs = torch.softmax(logits, dim=-1)[0]
    pred = int(probs.argmax().item())
    conf = float(probs[pred].item())
    label_ru = sentiment_display_name(pred, config)
    return label_ru, pred, conf


def emotions_from_logits(
    logits: torch.Tensor,
    label_names: list[str],
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    probs = torch.sigmoid(logits[0])
    above = []
    for i, p in enumerate(probs.tolist()):
        if p >= threshold and i < len(label_names):
            above.append((label_names[i], float(p)))
    above.sort(key=lambda x: -x[1])
    ranked = sorted(
        [(label_names[i], float(probs[i].item())) for i in range(min(len(label_names), len(probs)))],
        key=lambda x: -x[1],
    )[:top_k]
    return {"above_threshold": above, "top_k": ranked}


def forward_classification(
    text: str,
    *,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    with_hidden_states: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, torch.Tensor | None]:
    """
    Один forward классификатора. При with_hidden_states=True возвращает
    logits, hidden_states (кортеж по слоям), attention_mask.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    attention_mask = inputs["attention_mask"]
    with torch.inference_mode():
        if with_hidden_states:
            out = model(**inputs, output_hidden_states=True)
            return out.logits, out.hidden_states, attention_mask
        logits = model(**inputs).logits
        return logits, None, attention_mask


def hidden_states_to_layer_matrix(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> np.ndarray:
    """
    Mean pooling по токенам для каждого слоя → матрица (num_layers, hidden_dim).
    """
    rows: list[np.ndarray] = []
    for h in hidden_states:
        pooled = mean_pooling(h, attention_mask)  # (1, dim)
        rows.append(pooled[0].detach().float().cpu().numpy())
    return np.stack(rows, axis=0)


def predict_sentiment(
    text: str,
    *,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> tuple[str, int, float]:
    logits, _, _ = forward_classification(
        text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        with_hidden_states=False,
    )
    return sentiment_from_logits(logits, model.config)


def emotion_label_names(num_labels: int, names_from_dataset: list[str]) -> list[str]:
    if len(names_from_dataset) >= num_labels:
        return list(names_from_dataset[:num_labels])
    padded = list(names_from_dataset)
    padded.extend(f"эмоция_{i}" for i in range(len(padded), num_labels))
    return padded


def predict_emotions(
    text: str,
    *,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    label_names: list[str],
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    logits, _, _ = forward_classification(
        text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        with_hidden_states=False,
    )
    return emotions_from_logits(logits, label_names, threshold=threshold, top_k=top_k)
