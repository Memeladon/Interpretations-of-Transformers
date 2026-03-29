from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

ProblemType = Literal["single_label", "multi_label"]


def _compute_num_labels_single(labels: list[int]) -> int:
    return int(max(labels)) + 1 if labels else 1


def _labels_to_multihot(label_rows: list[list[int]], num_labels: int) -> np.ndarray:
    y = np.zeros((len(label_rows), num_labels), dtype=np.float32)
    for i, labs in enumerate(label_rows):
        for j in labs:
            if 0 <= j < num_labels:
                y[i, j] = 1.0
    return y


def _infer_num_labels_multilabel(label_rows: list[list[int]]) -> int:
    flat = [j for row in label_rows for j in row]
    return int(max(flat)) + 1 if flat else 1


def _tokenize_texts(
    tokenizer,
    texts: list[str],
    max_length: int,
) -> dict[str, Any]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def finetune_classifier(
    *,
    base_model_name: str,
    train_texts: list[str],
    train_labels: Any,
    output_dir: str | Path,
    problem_type: ProblemType,
    seed: int,
    max_length: int,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: float,
    train_batch_size: int,
    eval_batch_size: int,
    skip_if_exists: bool,
) -> Path:
    """
    Дообучение под тональность (single_label) или стиль/emotion (multi_label).
    Сохраняет чекпоинт совместимый с AutoModelForSequenceClassification.
    """
    out = Path(output_dir)
    has_ckpt = (out / "config.json").exists() and (
        (out / "pytorch_model.bin").exists() or (out / "model.safetensors").exists()
    )
    if skip_if_exists and has_ckpt:
        logger.info("finetune: skip (checkpoint exists): %s", out.resolve())
        return out

    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if problem_type == "single_label":
        y = np.array(train_labels, dtype=np.int64)
        num_labels = _compute_num_labels_single(y.tolist())
        label_tensor = y
        stratify = y
    else:
        label_rows: list[list[int]] = [list(row) if isinstance(row, list) else [int(row)] for row in train_labels]
        num_labels = _infer_num_labels_multilabel(label_rows)
        label_tensor = _labels_to_multihot(label_rows, num_labels)
        stratify = None

    idx = np.arange(len(train_texts))
    tr_idx, va_idx = train_test_split(idx, test_size=0.1, random_state=seed, stratify=stratify)

    tr_texts = [train_texts[i] for i in tr_idx]
    va_texts = [train_texts[i] for i in va_idx]

    tr_y = label_tensor[tr_idx]
    va_y = label_tensor[va_idx]

    train_ds = Dataset.from_dict({"text": tr_texts, "labels": tr_y.tolist()})
    val_ds = Dataset.from_dict({"text": va_texts, "labels": va_y.tolist()})

    def _tok(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        enc["labels"] = batch["labels"]
        return enc

    train_ds = train_ds.map(_tok, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(_tok, batched=True, remove_columns=["text"])

    hf_problem = "single_label_classification" if problem_type == "single_label" else "multi_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        problem_type=hf_problem,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _metrics(eval_pred):
        logits, labels = eval_pred
        if problem_type == "single_label":
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
            }
        probs = 1 / (1 + np.exp(-logits))
        pred_mult = (probs > 0.5).astype(int)
        y_true = np.array(labels)
        return {
            "f1_micro": float(f1_score(y_true, pred_mult, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, pred_mult, average="macro", zero_division=0)),
        }

    args = TrainingArguments(
        output_dir=str(out / "trainer_tmp"),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        seed=int(seed),
        save_strategy="no",
        logging_steps=50,
        eval_strategy="epoch",
        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_metrics,
    )

    logger.info("finetune: start -> %s (problem=%s, n_labels=%s)", out.resolve(), problem_type, num_labels)
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("finetune: saved %s", out.resolve())

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out
