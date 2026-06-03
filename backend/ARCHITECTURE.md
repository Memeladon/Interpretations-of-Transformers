# Архитектура backend

Система интерпретации эмбеддингов трансформерных языковых моделей на уровне **стиля** и **семантики**. Реализация соответствует модульной структуре ТЗ: независимые этапы, YAML-конфигурация, каталоги `data/`, `artifacts/`, `reports/`, `runs/`, `logs/`, `ui/`.

## Структура каталогов (ТЗ)

```
backend/
  configs/              # experiment.yaml — seed, модели, треки, probing, decomposition
  data/
    raw/                # сырые данные (резерв)
    processed/          # *_records.jsonl после нормализации HF
    splits/             # train/val/test индексы по задачам
  artifacts/
    finetuned/          # чекпоинты tone/style
    analysis/           # эмбеддинги, pipeline_summary.json, concept_directions.json
    runs/<run_id>/      # артефакты конкретного запуска (при run_pipeline)
  src/                  # библиотека
  reports/              # layer_profiles.json, intervention_summary.json, …
  ui/                   # интерактивный анализ текста
  logs/                 # run_YYYYMMDD_HHMMSS.log
  runs/                 # manifest.json + копия конфига на запуск
  experiments/          # entrypoint-скрипты этапов
```

## Точки входа (независимые этапы)

| Этап pipeline | Скрипт | Результат |
|---------------|--------|-----------|
| Подготовка данных | `experiments/prepare_data.py` | `data/processed/<track>_records.jsonl` |
| Splits | `experiments/build_splits.py` | `data/splits/<track>/<task>_split.json` |
| Hidden states + embeddings + probing + concept + intervention | `experiments/run_pipeline.py` (или `run_exp.py`) | `artifacts/.../analysis/...` |
| Устойчивость | `experiments/analyze_robustness.py` | `artifacts/robustness.json` |
| Отчёты | `experiments/generate_reports.py` | `reports/*.json` |
| Интерактив | `ui/interactive_tone_style.py` | консоль + matplotlib attention |

Каждый скрипт принимает `--config` (YAML/JSON), пишет лог в `logs/`, при необходимости создаёт запись в `runs/<run_id>/manifest.json`.

Полный последовательный pipeline: `uv run experiments/run_pipeline.py` (из каталога `backend`).

## Поток `run_pipeline`

1. **Runtime** — потоки BLAS/PyTorch/sklearn (`runtime` в конфиге).
2. **prepare_data** — `load_track_datasets` → JSONL в `data/processed/`.
3. **build_splits** — train/val/test по задачам → `data/splits/`.
4. **Fine-tuning** (опционально) — головы tone/style → `artifacts/finetuned/`.
5. **Анализ по трекам** — для каждой задачи/модели/уровня: `run_embedding_pipeline`:
   - encode + агрегация (text/sentence/token);
   - сохранение `.npy`;
   - **probing** → layer-wise профиль;
   - **concept directions** (PCA, probe_directions) → `concept_directions.json`;
   - **interventions** (проекция / null space);
   - `pipeline_summary.json`, `task_meta.json`.
6. **robustness** — bootstrap по seed на сохранённых эмбеддингах.
7. **reports** — агрегация в `reports/`.

## Модули `src/`

| Модуль | Назначение |
|--------|------------|
| `paths.py` | Корневые пути проекта |
| `experiment_config.py` | Загрузка YAML/JSON |
| `run_context.py` | Идентификатор запуска, manifest |
| `data_pipeline.py` | Подготовка processed-записей |
| `data_splits.py` | Train/val/test splits |
| `experiment_runner.py` | Fine-tuning + анализ по трекам |
| `datasets_load.py` | HF → единая схема записей |
| `embeddings/pipeline.py` | Encode, probing, concept, intervention (в т.ч. отдельные `run_*_stage`) |
| `probing.py` | Линейные зонды по слоям |
| `robustness.py` | Bootstrap-устойчивость probing |
| `reports.py` | Сводные отчёты |
| `tone_style_inference.py`, `attention_layer_viz.py` | Интерактив |

## Конфигурация

Декларативные файлы в `configs/` (см. `configs/CONFIG.md`):

- **`experiment.yaml`** — run id, модель (name/path), tokenization, layers, pooling, normalization, probing, interventions, device, artifacts
- **`datasets.yaml`** — документация HF-источников; опциональные file-источники через `data.sources.tracks`

Загрузка: `load_experiment_config()` → `resolve_experiment_config()` (нормализация + legacy-ключи).

## Схема данных (ТЗ)

Каждый пример: `text`, `y_style`, `y_semantic`, `metadata` (source, domain, language, split, pair_id, contrast_type, preprocessing_version, text_length).

Legacy-поля (`task_name`, `track`, `label`, …) сохраняются для совместимости с probing/fine-tuning.

## Подготовка данных

- Загрузка: HF (по умолчанию), CSV, JSON, JSONL, TXT — `src/data_loaders.py`
- Препроцессинг: очистка, нормализация пробелов, дедуп, фильтр пустых, лимит длины, балансировка — `src/data_preprocessing.py`
- Splits: `train.jsonl`, `val.jsonl`, `test.jsonl` + `split_manifest.json` — переиспользуются probing и fine-tuning
- **Leakage protocol** — `src/leakage_protocol.py`: пересечение текстов / pair_id между partition; `fail_on_violation` в конфиге

## Схема нормализованной записи

См. поле `track` ∈ {semantic, tone, style}, `task_type`, `label` / `labels`, `text`, `text_pair` — как в `datasets_load.py` и прежней документации.

## Ограничения

- Pair-задачи: в encode передаётся только `text` (не `text_pair`).
- Multi-label emotion: probing по списку `labels` при multi_label; иначе по `label`.

## Окружение

Windows + CPU: см. предупреждения в `experiment_runner.resolve_runtime`. Зависимости: `pyproject.toml` (`uv sync`).
