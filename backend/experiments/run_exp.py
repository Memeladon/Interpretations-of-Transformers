"""
Пакетный эксперимент
  uv run experiments/run_pipeline.py
Запуск из каталога backend.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from experiments.run_pipeline import main

    sys.exit(main() or 0)
