# Архитектура backend

Backend воспроизводит эксперименты по интерпретации трансформеров: загрузка нормализованных датасетов по **трекам** (semantic / tone / style), опциональный **fine-tuning** классификаторов под тон и стиль, извлечение **скрытых состояний** по слоям, **линейный probing**, **разложения** (PCA, направления пробы) и **интервенции** (проецирование / null space) с повторным probing.

## Точка входа

- Основной сценарий: `experiments/run_exp.py` (запуск из каталога `backend`, чтобы корректно резолвились пути к конфигу и `artifacts/`).
- Конфиг: `experiments/experiment_config.json` (загрузка через `src/experiment_config.load_experiment_config`).
- Логи: `artifacts/logs/run_YYYYMMDD_HHMMSS.log`; при старте включается `faulthandler` на тот же файл (на случай нативных падений).

## Поток выполнения (`run_exp`)

1. **Runtime** — до импорта `torch` на Windows выставляются лимиты потоков BLAS/OpenMP (`OMP_NUM_THREADS` и др. по умолчанию в `1`), чтобы снизить риск access violation при длинном CPU-forward. После загрузки конфига: `torch.set_num_threads`, `torch.set_num_interop_threads`, `TOKENIZERS_PARALLELISM`, `probing_n_jobs` для sklearn.
2. **Датасеты** — для каждого включённого трека вызывается `load_track_datasets` (`src/datasets_load.py`) с `cache_dir`, `dataset_limit_per_source`, `seed`.
3. **Fine-tuning** (если `finetuning.enabled`) — для каждой включённой модели: `finetune_classifier` (`src/finetuning.py`) на данных треков tone и style; чекпоинты в `artifacts/finetuned/<family>/tone|style/`.
4. **Анализ по трекам** — для semantic без чекпоинтов; для tone/style передаётся словарь `finetuned_by_family` (может быть пустым, тогда в лог пишется предупреждение о базовом backbone).
5. Внутри трека: по каждой **задаче** (`task_name`) и каждой **семейству модели** (`bert` / `gpt` / `llama`) — загрузка весов `load_language_model`, затем по каждому **уровню агрегации** из `levels` — `run_embedding_pipeline`.

## Конфигурация (ключи)

| Область | Назначение |
|--------|------------|
| `seed` | Воспроизводимость (numpy, torch, сэмплирование датасетов). |
| `cache_dir` | Кэш Hugging Face Datasets. |
| `dataset_limit_per_source` | Лимит строк на исходный HF-датасет после нормализации. |
| `models` | Флаги семейств; хотя бы одно должно быть `true`. |
| `finetuning` | Параметры обучения головы (`epochs`, LR, batch sizes, `fp16`/`bf16`, `skip_if_exists`, …). |
| `levels` | Уровни эмбеддингов: `text`, `sentence`, `token`. |
| `text_strategy` | Стратегия пулинга для агрегации (например `mean`) — см. `aggregation.py`. |
| `batch_size`, `max_length` | Базовые параметры encode; переопределяются `runtime_overrides` по семейству. |
| `runtime` | `cpu_threads` (`0`/null → авто; на Windows без CUDA авто принудительно **1** из-за стабильности PyTorch), `torch_interop_threads`, `tokenizers_parallelism`, `probing_n_jobs` (`-1` → все ядра в `LogisticRegression` / части sklearn). |
| `decomposition` | `pca_components`, `enabled_methods`, `interventions`, `drop_components`. |
| `tracks` | Включение трека и список `tasks` (имена должны совпадать с мапперами в `datasets_load`). |

## Структура каталогов

```
backend/
  experiments/
    run_exp.py              # главный сценарий
    experiment_config.json
    inspect_datasets.py     # отладка: сырые/нормализованные записи, опционально токены
  src/
    datasets_load.py        # треки, HF-спеки, нормализация в записи
    experiment_config.py    # load_experiment_config
    experiment_logging.py   # цветной консольный лог + файл
    finetuning.py           # fine-tune sequence classification
    utils.py
    probing.py              # train_probes_by_layer, метрики classification/regression
    linear_model.py         # вспомогательный sklearn (не ядро пайплайна)
    language_models/
      loader.py             # MODEL_REGISTRY, load_language_model, base_model_name_for_family
    embeddings/
      extractor.py            # EmbeddingExtractor.encode: forward + агрегация по уровню
      aggregation.py        # aggregate_layer (text/sentence/token)
      pipeline.py           # run_embedding_pipeline: save .npy, probing, decomposition, intervention
  artifacts/                # создаётся при запуске (gitignore)
    logs/
    finetuned/              # при enabled finetuning
    analysis/<track>/<task>/<family>/<level>/
      pipeline_summary.json
      <level>_<strategy>_layer_*.npy
```

## Модели (`language_models`)

Реестр `MODEL_REGISTRY` в `loader.py`: соответствие семейства → имя чекпоинта на Hugging Face. Без `finetuned_checkpoint` грузится `AutoModel`; с чекпоинтом — `AutoModelForSequenceClassification` (после fine-tune), с `output_hidden_states=True` при encode.

## Пайплайн эмбеддингов (`embeddings/pipeline.py`)

1. **Encode** — батчами через `EmbeddingExtractor`: для `level != None` сразу агрегируются представления по слоям (экономия памяти против хранения полных `(N, seq, hidden)` по всем слоям).
2. **Сохранение** — `save_layer_embeddings` → `.npy` в каталог уровня.
3. **Probing** — по матрицам слоёв: `train_probes_by_layer` (`probing.py`); для multi-label — метрика `f1_macro`, иначе `accuracy`; для регрессии — `neg_mse`.
4. **Decomposition** — на последнем слое: опционально PCA, направления линейной пробы (`probe_directions`), при необходимости — подготовка для `null_space`.
5. **Intervention** — проецирование первых `drop_components` компонент (PCA или направления пробы) или удаление в null space; после каждого варианта снова probing по одному «виртуальному» слою.
6. **Итог** — `pipeline_summary.json` (пути к эмбеддингам, скоры probing, сериализованное разложение, скоры после интервенций).

## Треки и источники данных

| Трек | Задачи (пример из конфига) | HF-датасеты (спеки в коде) |
|------|----------------------------|----------------------------|
| semantic | `semantic_similarity`, `paraphrase` | `ai-forever/ru-stsbenchmark-sts`, `MilyaShams/qqp-ru_10k` |
| tone | `sentiment` | `MonoHime/ru_sentiment_dataset` |
| style | `emotion` | `seara/ru_go_emotions` (config `raw`) |

Добавление нового источника: новый элемент в `SEMANTIC_SPECS` / `TONE_SPECS` / `STYLE_SPECS` с функцией-маппером, возвращающей запись в общей схеме (ниже).

## Схема нормализованной записи (датасет)

Все строки после `datasets_load` унифицированы. Минимально осмысловые поля:

```json
{
  "id": "str",
  "source_id": "int",
  "source_name": "str",
  "source_dataset": "str",
  "split": "str",
  "task_name": "str",
  "track": "semantic | tone | style",
  "task_group": "semantic | supervised",
  "task_type": "classification | regression",
  "label_type": "single_label | multi_label | regression",
  "text": "str",
  "text_pair": "str | null",
  "is_pair_task": "bool",
  "label": "int | float | null",
  "labels": "list[int] | null"
}
```

- Для **парных** задач (STS, QQP) заполняются `text` и `text_pair`.
- В **`run_exp`** в модель на этапе encode передаётся **только** список полей `text` (второе предложение в `text_pair` в forward **не склеивается** — это ограничение текущего сценария; см. подсказку в `inspect_datasets.py`).

## Замечания по окружению

- **Windows + CPU**: при авто-выборе `cpu_threads` принудительно используется 1 поток PyTorch; для скорости — CUDA или явное увеличение `cpu_threads` с пониманием риска.
- **`probing_n_jobs > 1`**: sklearn использует joblib/loky; при завершении процесса на Windows возможны предупреждения `resource_tracker` / временные папки в `%TEMP%` — обычно не означают порчи логики эксперимента; при проблемах можно выставить `probing_n_jobs: 1`.

## Вспомогательные сценарии

- `experiments/inspect_datasets.py` — просмотр спецификаций треков, примеров нормализованных строк и опционально выхода токенизатора (`--show-tokens`, `--model-family`).
