"""
Интерактив: ввод текста в консоль → тональность (tone) и стиль/эмоции (style) дообученными головами,
затем визуализация mean-pooled эмбеддингов по каждому слою (отдельно для чекпоинта tone и style).

Запуск из каталога backend (как run_exp):
  uv run experiments/interactive_tone_style.py
  uv run experiments/interactive_tone_style.py --family gpt
  uv run experiments/interactive_tone_style.py --no-viz

Нужны чекпоинты после fine-tuning:
  artifacts/finetuned/<family>/tone
  artifacts/finetuned/<family>/style
См. experiment_config.json → finetuning.enabled и run_exp.
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
    for _k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_k, "1")

import torch

from src.datasets_load import GO_EMOTIONS_BINARY_COLUMNS
from src.experiment_config import load_experiment_config
from src.layer_embedding_viz import plot_tone_style_layer_embeddings
from src.tone_style_inference import (
    checkpoint_dir_ready,
    emotion_label_names,
    emotions_from_logits,
    forward_classification,
    hidden_states_to_layer_matrix,
    load_classification_head,
    sentiment_from_logits,
)

log = logging.getLogger("interactive_tone_style")


def _max_length_for_family(cfg: dict, family: str) -> int:
    m = int(cfg["max_length"])
    overrides = (cfg.get("runtime_overrides") or {}).get(family, {})
    if "max_length" in overrides:
        m = int(overrides["max_length"])
    return m


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Интерактив: тональность и эмоции дообученной моделью.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/experiment_config.json"),
        help="Путь к experiment_config.json",
    )
    p.add_argument(
        "--family",
        choices=("bert", "gpt", "llama"),
        default="bert",
        help="Семейство модели (каталог чекпоинтов finetuned/<family>/…)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Порог сигмоиды для multi-label эмоций",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (по умолчанию: cuda если доступна)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Меньше служебных сообщений")
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Не открывать окно matplotlib с эмбеддингами по слоям",
    )
    p.add_argument(
        "--save-viz-dir",
        type=Path,
        default=None,
        help="Сохранять PNG визуализаций в каталог (имя файла с timestamp)",
    )
    p.add_argument(
        "--heatmap-dims",
        type=int,
        default=48,
        help="Сколько первых размерностей показывать на теплокарте слой×dim",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    cfg_path = args.config
    if not cfg_path.is_file():
        log.error("Не найден конфиг: %s", cfg_path.resolve())
        sys.exit(1)

    cfg = load_experiment_config(cfg_path)
    ft = cfg.get("finetuning") or {}
    fin_root = Path(ft.get("output_dir", "artifacts/finetuned"))
    family = args.family
    tone_dir = fin_root / family / "tone"
    style_dir = fin_root / family / "style"

    if not checkpoint_dir_ready(tone_dir):
        log.error(
            "Нет чекпоинта тональности: %s\n"
            "Включите finetuning в конфиге и выполните run_exp, либо укажите корректный output_dir.",
            tone_dir.resolve(),
        )
        sys.exit(1)
    if not checkpoint_dir_ready(style_dir):
        log.error(
            "Нет чекпоинта стиля (эмоции): %s",
            style_dir.resolve(),
        )
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_length = _max_length_for_family(cfg, family)

    log.info("Устройство: %s", device)
    log.info("Семейство: %s, max_length=%s", family, max_length)
    log.info("Загрузка головы тональности: %s", tone_dir.resolve())
    tok_tone, model_tone = load_classification_head(tone_dir, device)
    log.info("Загрузка головы эмоций: %s", style_dir.resolve())
    tok_style, model_style = load_classification_head(style_dir, device)

    n_emotion = int(getattr(model_style.config, "num_labels", 0))
    emotion_names = emotion_label_names(n_emotion, list(GO_EMOTIONS_BINARY_COLUMNS))

    print()
    print("Введите текст и нажмите Enter. Пустая строка или «quit» / «exit» — выход.")
    print("—" * 60)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            print("Выход.")
            break

        logits_t, hidden_t, mask_t = forward_classification(
            line,
            tokenizer=tok_tone,
            model=model_tone,
            device=device,
            max_length=max_length,
            with_hidden_states=True,
        )
        assert hidden_t is not None and mask_t is not None
        tone_label, tone_id, tone_p = sentiment_from_logits(logits_t, model_tone.config)
        tone_layers = hidden_states_to_layer_matrix(hidden_t, mask_t)

        logits_s, hidden_s, mask_s = forward_classification(
            line,
            tokenizer=tok_style,
            model=model_style,
            device=device,
            max_length=max_length,
            with_hidden_states=True,
        )
        assert hidden_s is not None and mask_s is not None
        emo = emotions_from_logits(
            logits_s,
            emotion_names,
            threshold=args.threshold,
            top_k=5,
        )
        style_layers = hidden_states_to_layer_matrix(hidden_s, mask_s)

        print()
        print(f"  Тональность: {tone_label} (id={tone_id}, уверенность {tone_p:.3f})")
        print(
            f"  Слои (тональность): {tone_layers.shape[0]} слоёв, размерность {tone_layers.shape[1]}"
        )
        if emo["above_threshold"]:
            joined = ", ".join(f"{name} ({p:.3f})" for name, p in emo["above_threshold"])
            print(f"  Эмоции (≥ {args.threshold:g}): {joined}")
        else:
            print(f"  Эмоции (≥ {args.threshold:g}): нет; топ по вероятности:")
            for name, p in emo["top_k"][:5]:
                print(f"    · {name}: {p:.3f}")
        print(
            f"  Слои (стиль): {style_layers.shape[0]} слоёв, размерность {style_layers.shape[1]}"
        )

        save_path = None
        if args.save_viz_dir is not None:
            from datetime import datetime

            args.save_viz_dir.mkdir(parents=True, exist_ok=True)
            safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = args.save_viz_dir / f"layers_{safe_ts}.png"

        if not args.no_viz:
            print("  Открываю график эмбеддингов по слоям (закройте окно — вернётесь к вводу).")
            plot_tone_style_layer_embeddings(
                tone_layers,
                style_layers,
                text_preview=line,
                show=True,
                save_path=save_path,
                heatmap_dims=args.heatmap_dims,
            )
        elif save_path is not None:
            plot_tone_style_layer_embeddings(
                tone_layers,
                style_layers,
                text_preview=line,
                show=False,
                save_path=save_path,
                heatmap_dims=args.heatmap_dims,
            )
            print(f"  График сохранён: {save_path.resolve()}")

        print("—" * 60)


if __name__ == "__main__":
    main()
