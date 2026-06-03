"""
Интерактивный анализ пользовательского текста (тональность + стиль/эмоции + attention по слоям).

Запуск из каталога backend:
  uv run ui/interactive_tone_style.py
  uv run ui/interactive_tone_style.py --family bert --save-viz-dir artifacts/interactive_viz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

if os.name == "nt":
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

import torch

from src.attention_layer_viz import plot_attention_per_layer_tone_style
from src.datasets_load import GO_EMOTIONS_BINARY_COLUMNS
from src.experiment_config import load_experiment_config
from src.paths import ProjectPaths
from src.tone_style_inference import (
    checkpoint_dir_ready,
    emotion_label_names,
    emotions_from_logits,
    forward_classifier_with_attentions,
    load_classification_head,
    sentiment_from_logits,
)

log = logging.getLogger("ui.interactive")


def _max_length_for_family(cfg: dict, family: str) -> int:
    m = int(cfg["max_length"])
    overrides = (cfg.get("runtime_overrides") or {}).get(family, {})
    if "max_length" in overrides:
        m = int(overrides["max_length"])
    return m


def _parse_args() -> argparse.Namespace:
    paths = ProjectPaths.from_root(_BACKEND_ROOT)
    p = argparse.ArgumentParser(description="Интерактивный анализ текста (тон + стиль).")
    p.add_argument("--config", type=Path, default=paths.default_config_path())
    p.add_argument("--family", choices=("bert", "gpt", "llama"), default="bert")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--save-viz-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    if not args.config.is_file():
        log.error("Не найден конфиг: %s", args.config.resolve())
        sys.exit(1)

    cfg = load_experiment_config(args.config)
    ft = cfg.get("finetuning") or {}
    fin_root = Path(ft.get("output_dir", "artifacts/finetuned"))
    family = args.family
    tone_dir = fin_root / family / "tone"
    style_dir = fin_root / family / "style"

    if not checkpoint_dir_ready(tone_dir):
        log.error("Нет чекпоинта тональности: %s (сначала run_pipeline / fine-tuning)", tone_dir.resolve())
        sys.exit(1)
    if not checkpoint_dir_ready(style_dir):
        log.error("Нет чекпоинта стиля: %s", style_dir.resolve())
        sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    max_length = _max_length_for_family(cfg, family)

    log.info("Устройство: %s | %s | max_length=%s", device, family, max_length)
    tok_tone, model_tone = load_classification_head(tone_dir, device, attn_implementation="eager")
    tok_style, model_style = load_classification_head(style_dir, device, attn_implementation="eager")
    n_emotion = int(getattr(model_style.config, "num_labels", 0))
    emotion_names = emotion_label_names(n_emotion, list(GO_EMOTIONS_BINARY_COLUMNS))

    print("\nВведите текст (пустая строка / quit — выход).\n" + "—" * 60)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            print("Выход.")
            break

        try:
            logits_t, attentions_t, ids_t, mask_t = forward_classifier_with_attentions(
                line, tokenizer=tok_tone, model=model_tone, device=device, max_length=max_length
            )
            logits_s, attentions_s, ids_s, mask_s = forward_classifier_with_attentions(
                line, tokenizer=tok_style, model=model_style, device=device, max_length=max_length
            )
        except RuntimeError as exc:
            print(f"  Ошибка forward: {exc}\n" + "—" * 60)
            continue

        tone_label, tone_id, tone_p = sentiment_from_logits(logits_t, model_tone.config)
        emo = emotions_from_logits(logits_s, emotion_names, threshold=args.threshold, top_k=5)

        print(f"\n  Тональность: {tone_label} (id={tone_id}, {tone_p:.3f})")
        if emo["above_threshold"]:
            print("  Эмоции:", ", ".join(f"{n} ({p:.3f})" for n, p in emo["above_threshold"]))
        else:
            print("  Эмоции (топ):")
            for name, p in emo["top_k"][:5]:
                print(f"    · {name}: {p:.3f}")

        save_dir = None
        if args.save_viz_dir is not None:
            from datetime import datetime

            args.save_viz_dir.mkdir(parents=True, exist_ok=True)
            save_dir = args.save_viz_dir / f"attention_{datetime.now():%Y%m%d_%H%M%S}"

        want_show = not args.no_viz
        if want_show or save_dir is not None:
            note, saved_paths = plot_attention_per_layer_tone_style(
                attentions_t,
                attentions_s,
                input_ids_tone=ids_t,
                mask_tone=mask_t,
                tok_tone=tok_tone,
                input_ids_style=ids_s,
                mask_style=mask_s,
                tok_style=tok_style,
                text_preview=line,
                show=want_show,
                save_dir=save_dir,
            )
            print(f"  {note}")
            if saved_paths:
                print(f"  Сохранено: {saved_paths[0].parent.resolve()}")
        print("—" * 60)


if __name__ == "__main__":
    main()
