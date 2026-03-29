from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import colorlog


def setup_experiment_logging(
    log_dir: Path,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """
    Консоль: цвет по уровню, короткое имя модуля, время. Файл: без ANSI, фиксированные колонки.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_fmt = colorlog.ColoredFormatter(
        fmt=(
            "%(cyan)s%(asctime)s%(reset)s  "
            "%(log_color)s%(levelname)-7s%(reset)s  "
            "%(blue)s%(shortname)-28s%(reset)s | "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "white",
            "INFO": "bold_green",
            "WARNING": "bold_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_white,bg_red",
        },
        style="%",
    )

    class _ShortNameFilter(logging.Filter):
        """Читаемое имя: без префикса src., последние два сегмента (embeddings.pipeline)."""

        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name
            if name.startswith("src."):
                name = name[4:]
            parts = name.split(".")
            if len(parts) >= 2:
                short = f"{parts[-2]}.{parts[-1]}"
            else:
                short = parts[0]
            max_w = 28
            if len(short) > max_w:
                short = short[: max_w - 1] + "…"
            record.shortname = short
            return True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_fmt)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_fmt)
    console_handler.addFilter(_ShortNameFilter())

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)

    return log_path


def log_section(logger: logging.Logger, title: str, *, width: int = 76) -> None:
    """Крупный визуальный разделитель в логе (консоль и файл)."""
    bar = "━" * width
    logger.info("")
    logger.info(bar)
    logger.info("  ▶ %s", title)
    logger.info(bar)


def log_ruler(logger: logging.Logger, char: str = "·", width: int = 72) -> None:
    logger.info(char * width)


def _sort_layer_keys(keys: list[str]) -> list[str]:
    def _key(k: str) -> tuple[int, str]:
        try:
            return (0, f"{int(k):04d}")
        except ValueError:
            return (1, k)

    return sorted(keys, key=lambda k: _key(str(k)))


def log_layer_scores(
    logger: logging.Logger,
    title: str,
    scores: dict[str, float],
    *,
    metric: str = "score",
) -> None:
    """Табличный вывод метрик по слоям (удобно смотреть в консоли и в .log)."""
    if not scores:
        return
    logger.info("  %s (%s):", title, metric)
    ordered = _sort_layer_keys(list(scores))
    best_k = max(ordered, key=lambda k: scores[k])
    for k in ordered:
        mark = "  *" if k == best_k else "   "
        logger.info("%s layer %3s  %-10s %.4f", mark, k, metric, scores[k])
    logger.info("  лучший слой: %s = %.4f", best_k, scores[best_k])
