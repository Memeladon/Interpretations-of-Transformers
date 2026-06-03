# Система интерпретации эмбеддингов трансформерных языковых моделей

Исследовательский Python-проект для анализа скрытых представлений трансформеров: линейный probing, layer-wise профили, concept directions, representation interventions и интерактивный разбор текста.

Код и документация — в каталоге [`backend/`](backend/).

## Структура (соответствие ТЗ)

```
backend/
  configs/          # YAML-конфигурации экспериментов
  data/             # raw, processed, splits
  artifacts/        # эмбеддинги, модели, результаты анализа
  src/              # библиотека
  reports/          # автогенерируемые отчёты
  ui/               # интерактивный анализ текста
  logs/             # логи запусков
  runs/             # метаданные отдельных запусков (manifest + копия конфига)
  experiments/      # entrypoint-скрипты этапов pipeline
```

## Быстрый старт

Из корня репозитория:

```bash
uv sync
uv run backend/experiments/run_pipeline.py
```

Если вы уже находитесь в каталоге `backend`, используйте путь без префикса `backend/`:

```bash
uv sync
uv run experiments/run_pipeline.py
```

Отдельные этапы (независимые entrypoint'ы):

| Этап | Команда |
|------|---------|
| Подготовка данных | `uv run backend/experiments/prepare_data.py` |
| Splits | `uv run backend/experiments/build_splits.py` |
| Полный анализ | `uv run backend/experiments/run_pipeline.py` |
| Устойчивость | `uv run backend/experiments/analyze_robustness.py` |
| Отчёты | `uv run backend/experiments/generate_reports.py` |
| Интерактив | `uv run backend/ui/interactive_tone_style.py` |

Аналогично из каталога `backend`: `uv run experiments/<script>.py`, для UI — `uv run ui/interactive_tone_style.py`.

Конфиг по умолчанию: `configs/experiment.yaml`. Подробнее — [`backend/ARCHITECTURE.md`](backend/ARCHITECTURE.md).

### Офлайн / нестабильный HuggingFace

Если скачивание с HuggingFace падает по таймауту, pipeline автоматически подставит последний локальный дамп из `data/raw/dataset_dump/` (если он есть). Чтобы не перекачивать данные повторно:

```bash
# из backend/
uv run experiments/run_pipeline.py --skip-prepare-data
```

Принудительная пересборка `data/processed/`:

```bash
uv run experiments/run_pipeline.py --force-prepare-data
```
