"""
Просмотр датасетов: сырой HF -> нормализованная запись -> (опционально) батч токенизатора как вход модели.

Запуск из каталога backend:
  uv run experiments/inspect_datasets.py
  uv run experiments/inspect_datasets.py --track tone --n 5
  uv run experiments/inspect_datasets.py --track semantic --show-tokens --model-family bert
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from transformers import AutoTokenizer

from src.datasets_load import SEMANTIC_SPECS, STYLE_SPECS, TONE_SPECS, load_track_datasets
from src.experiment_config import load_experiment_config
from src.language_models import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("inspect_datasets")


def _pretty(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _specs_summary() -> None:
    log.info("=== Какие датасеты подключаются (по трекам) ===\n")
    for name, specs in ("semantic", SEMANTIC_SPECS), ("tone", TONE_SPECS), ("style", STYLE_SPECS):
        log.info("%s:", name)
        for s in specs:
            cfg = s.get("config_name")
            ds = s["dataset_name"]
            if cfg:
                log.info("  - %s  config=%r  split=%s", ds, cfg, s["split"])
            else:
                log.info("  - %s  split=%s", ds, s["split"])
        log.info("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect normalized dataset rows and tokenizer inputs.")
    parser.add_argument("--track", choices=("semantic", "tone", "style", "all"), default="all")
    parser.add_argument("--n", type=int, default=3, help="How many examples to print per track")
    parser.add_argument("--config", type=Path, default=Path("experiments/experiment_config.json"))
    parser.add_argument("--show-tokens", action="store_true", help="Show tokenizer batch for first example text")
    parser.add_argument(
        "--model-family",
        choices=tuple(MODEL_REGISTRY.keys()),
        default="bert",
        help="Tokenizer (and optional model) checkpoint: bert | gpt | llama",
    )
    args = parser.parse_args()

    limit = None
    cache_dir = ".cache/hf_datasets"
    seed = 42
    max_length = 256
    if args.config.exists():
        cfg = load_experiment_config(args.config)
        limit = cfg.get("dataset_limit_per_source")
        cache_dir = cfg.get("cache_dir", cache_dir)
        seed = cfg.get("seed", seed)
        max_length = cfg.get("max_length", max_length)
        log.info("Используется конфиг: %s", args.config.resolve())
    else:
        log.info("Конфиг не найден, берутся значения по умолчанию (limit=None).")

    _specs_summary()
    log.info("Параметры загрузки: cache_dir=%s  limit_per_source=%s  seed=%s\n", cache_dir, limit, seed)

    tracks: list[str] = list({"semantic", "tone", "style"} if args.track == "all" else {args.track})

    tokenizer = None
    if args.show_tokens:
        model_name = MODEL_REGISTRY[args.model_family]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Токенизатор: %s (family=%s)  max_length=%s\n", model_name, args.model_family, max_length)

    for track in tracks:
        rows = load_track_datasets(track, cache_dir=cache_dir, limit_per_dataset=limit, seed=seed)  # type: ignore[arg-type]
        log.info("======== track=%s  total_rows=%s ========", track, len(rows))
        for i, row in enumerate(rows[: args.n]):
            log.info("--- example %s / %s ---", i + 1, min(args.n, len(rows)))
            log.info("%s\n", _pretty(row))

        if args.show_tokens and rows and tokenizer is not None:
            text = rows[0]["text"]
            enc = tokenizer(
                [text],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            log.info("=== Токенизация первого примера (как в EmbeddingExtractor) ===")
            log.info("text (snippet): %s...", text[:200])
            for k, v in enc.items():
                if hasattr(v, "shape"):
                    log.info("  %s: shape=%s dtype=%s", k, tuple(v.shape), v.dtype)
            ids = enc["input_ids"][0].tolist()
            log.info("  input_ids (first 40): %s", ids[:40])
            tokens = tokenizer.convert_ids_to_tokens(ids[:40])
            log.info("  tokens (first 40): %s", tokens)
        log.info("")

    log.info(
        "Итог: для модели в батче всегда padding=max_length=%s, truncation=True; "
        "в run_exp/analysis в forward передаётся только поле text (пары text_pair сейчас не склеиваются).",
        max_length,
    )


if __name__ == "__main__":
    main()
