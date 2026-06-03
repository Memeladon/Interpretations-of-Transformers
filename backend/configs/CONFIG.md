# Конфигурационная система

Конфигурации хранятся в `configs/` отдельно от кода. Основной файл: **`experiment.yaml`**.

## Поля эксперимента (ТЗ)

| Поле | Секция YAML | Описание |
|------|-------------|----------|
| Идентификатор запуска | `run.id` | `null` → автоматически |
| Имя модели | `model.families.<fam>.name` | HF model id |
| Путь к модели | `model.families.<fam>.path` | Локальный путь (приоритет над name) |
| Токенизация | `tokenization.*` | max_length, padding, truncation |
| Слои | `layers.*` | mode: all / last_n / indices |
| Pooling | `pooling.strategy`, `pooling.levels` | mean, text/sentence/token |
| Нормировка | `normalization.probing_scaler` | standard / none |
| Probing | `probing.*` | test_size, n_jobs, use_predefined_splits |
| Интервенции | `interventions.*` | методы, drop_components |
| batch size | `batch_size` | |
| seed | `seed` | |
| device | `device` | auto / cuda / cpu |
| Пути артефактов | `artifacts.*` | root, processed, splits, reports, … |

## Схема примера (данные)

```json
{
  "text": "...",
  "y_style": null,
  "y_semantic": 1,
  "metadata": {
    "source": "MonoHime/ru_sentiment_dataset",
    "domain": "tone",
    "language": "ru",
    "split": "train",
    "pair_id": null,
    "contrast_type": null,
    "preprocessing_version": "1.0",
    "text_length": 42
  }
}
```

## Источники данных

- **huggingface** (по умолчанию) — как раньше, через `src/datasets_load.py`
- **csv / json / jsonl / txt** — `data.sources.tracks.<track>.type` + `path`

См. также `configs/datasets.yaml` для документирования HF-наборов.

## Splits и leakage

1. `prepare_data.py` → `data/processed/`
2. `build_splits.py` → `data/splits/<track>/<task>/train|val|test.jsonl` + `split_manifest.json`
3. Leakage protocol при сборке splits (дубликаты текста, pair_id между partition)

Все этапы pipeline используют одни и те же split-артефакты, если `probing.use_predefined_splits: true`.
