"""
Запрос: [CLS] → токены (энкодер) или последний токен → токены (causal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _format_tok(t: str) -> str:
    s = t.replace("Ġ", " ").replace("##", "").replace("▁", " ")
    return s if s.strip() else t


def _truncate_title(text: str, max_len: int = 72) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


MAX_TOKENS_SHOWN = 48


def salience_from_attention_layer(
    attn: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
) -> tuple[list[str], np.ndarray, str]:
    """
    attn: (batch, num_heads, seq, seq)
    Возвращает подписи токенов, вектор весов по позициям (ключи), описание запроса.
    """
    a = attn.float().mean(dim=1)[0].cpu()  # (seq, seq)
    mask = attention_mask[0].bool().cpu()
    seq_len = int(mask.sum().item())
    a = a[:seq_len, :seq_len]
    ids = input_ids[0, :seq_len].detach().cpu().numpy()

    cls_id = tokenizer.cls_token_id
    if cls_id is not None and int(ids[0]) == int(cls_id):
        sal = a[0, :].detach().cpu().numpy()
        desc = "[CLS] → токены"
    else:
        sal = a[seq_len - 1, :].detach().cpu().numpy()
        desc = "последний токен → токены"

    sal = np.asarray(sal, dtype=np.float64)
    sal = sal / (sal.sum() + 1e-9)

    raw = tokenizer.convert_ids_to_tokens(ids.tolist())
    tokens = [_format_tok(x) for x in raw]
    if len(tokens) > MAX_TOKENS_SHOWN:
        tokens = tokens[:MAX_TOKENS_SHOWN]
        sal = sal[:MAX_TOKENS_SHOWN]
        sal = sal / (sal.sum() + 1e-9)
    return tokens, sal, desc


def plot_attention_per_layer_tone_style(
    attentions_tone: tuple[torch.Tensor, ...],
    attentions_style: tuple[torch.Tensor, ...],
    *,
    input_ids_tone: torch.Tensor,
    mask_tone: torch.Tensor,
    tok_tone: Any,
    input_ids_style: torch.Tensor,
    mask_style: torch.Tensor,
    tok_style: Any,
    text_preview: str,
    show: bool = True,
    save_dir: Path | None = None,
) -> tuple[str, list[Path]]:
    """

    - ``show=True`` — последовательно открывается окно на каждый слой (закрыть → следующий).
    - ``save_dir`` — каталог; файлы ``layer_00.png``, ``layer_01.png``, …

    """
    n_t = len(attentions_tone)
    n_s = len(attentions_style)
    if n_t != n_s:
        raise ValueError(f"Число слоёв tone ({n_t}) и style ({n_s}) не совпадает")
    n_layers = n_t

    query_note = ""
    saved: list[Path] = []

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    preview = _truncate_title(text_preview)

    for layer_idx in range(n_layers):
        tokens_t, sal_t, desc_t = salience_from_attention_layer(
            attentions_tone[layer_idx], input_ids_tone, mask_tone, tok_tone
        )
        tokens_s, sal_s, desc_s = salience_from_attention_layer(
            attentions_style[layer_idx], input_ids_style, mask_style, tok_style
        )
        if layer_idx == 0:
            query_note = f"Запрос: {desc_t} (тональность) · {desc_s} (стиль)"

        ntok = max(len(tokens_t), len(tokens_s), 4)
        fig_h = min(22.0, max(4.2, 0.32 * ntok + 2.0))
        fig, axes = plt.subplots(1, 2, figsize=(13.5, fig_h))

        y_pos = np.arange(len(tokens_t))
        axes[0].barh(y_pos, sal_t, color="steelblue", height=0.65)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(tokens_t, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("вес внимания (среднее по головам)")
        axes[0].set_title("Тональность", fontsize=10)
        axes[0].set_xlim(0, max(0.05, float(sal_t.max()) * 1.15))

        y_pos_s = np.arange(len(tokens_s))
        axes[1].barh(y_pos_s, sal_s, color="darkorange", height=0.65)
        axes[1].set_yticks(y_pos_s)
        axes[1].set_yticklabels(tokens_s, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("вес внимания (среднее по головам)")
        axes[1].set_title("Стиль (эмоции)", fontsize=10)
        axes[1].set_xlim(0, max(0.05, float(sal_s.max()) * 1.15))

        fig.suptitle(
            f"Слой {layer_idx} (всего слоёв: {n_layers}) · Self-attention\n{preview}",
            fontsize=11,
        )
        fig.text(0.5, 0.02, query_note, ha="center", fontsize=8, style="italic")
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))

        if save_dir is not None:
            out = save_dir / f"layer_{layer_idx:02d}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            saved.append(out)

        if show:
            plt.show()
        plt.close(fig)

    return query_note, saved
