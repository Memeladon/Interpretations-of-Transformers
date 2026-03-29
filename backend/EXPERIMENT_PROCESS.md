# Текущий процесс выполнения эксперимента

Ниже описан **весь** пайплайн, который выполняется при запуске:
`uv run experiments/run_exp.py`

Описание опирается на текущие реализации:
- `experiments/experiment_config.json`
- `experiments/run_exp.py`
- `src/datasets_load.py`
- `src/language_models/loader.py`
- `src/embeddings/extractor.py`
- `src/embeddings/aggregation.py`
- `src/embeddings/pipeline.py`
- `src/probing.py`
- `src/experiment_logging.py`

---

## 0) Входы (что задаёт эксперимент)

### Конфиг
Читается из `experiments/experiment_config.json`. В текущем конфиге:
- `seed`: 42
- `cache_dir`: `.cache/hf_datasets`
- `dataset_limit_per_source`: 500
- `models`: какие модели включены (`bert`, `gpt`, `llama`)
- `levels`: какие уровни анализа будут выполнены (`text`, `sentence`, `token`)
- `text_strategy`: стратегия для уровня `text` (`mean`)
- `batch_size`: 8
- `max_length`: 256
- `decomposition.enabled_methods`: `["pca", "probe_directions", "null_space"]`
- `decomposition.interventions`: `["pca", "probe_directions", "null_space"]`
- `decomposition.drop_components`: 1
- `task_groups`:
  - `style`: `["sentiment", "emotion"]`
  - `semantic`: `["semantic_similarity", "paraphrase"]`

### Данные
Список датасетов берётся из `src/datasets_load.py` (`DATASET_SPECS`), и каждый датасет маппится в унифицированную схему записи.

---

## 1) Инициализация и логирование

Функция: `experiments/run_exp.py -> main()`

Порядок:
1. Создаётся лог-файл и настройка логирования:
   - `setup_experiment_logging(output_root / "logs")`
   - консоль: цветные уровни (ANSI)
   - файл: без ANSI
2. `set_seed(cfg["seed"])`:
   - задаёт seed для `numpy` и `torch` (CPU/GPU)

---

## 2) Загрузка и нормализация датасетов

Функция: `src/datasets_load.py -> load_all_datasets(...)`

### 2.1 Загрузка HF датасетов
Для каждого элемента `DATASET_SPECS` выполняется:
1. `load_dataset(dataset_name, name=config_name, split=split, cache_dir=cache_path)`
2. Проход по строкам датасета и построение унифицированных полей через соответствующий `mapper`.

### 2.2 Нормализация в единый формат записи
Каждая строка превращается в dict, содержащий (важное):
- `id`
- `source_id` (row_id внутри source)
- `source_name`, `source_dataset`, `split`
- `task_name`
- `task_group`: `style` или `semantic`
- `task_type`: например `classification` или `regression`
- `label_type`: `regression` / `single_label` / `multi_label`
- `text` и (если есть) `text_pair`
- `label`: **скалярная** метка для текущего probing-пайплайна (backward-compatible)
- `labels`: **lossless** список меток (особенно полезно для multi-label)

### 2.3 Фильтрация пустых текстов
Если `row["text"]` пустой — строка отбрасывается.

### 2.4 Сэмплинг `dataset_limit_per_source` без bias
Если limit задан:
- выбираются случайные строки в пределах каждого датасета
- случайность воспроизводима через `seed` (`random.Random(seed)`)

После накопления всех датасетов:
- делается `rng.shuffle(rows)`

---

## 3) Внешний цикл: группы задач -> задача -> модель -> уровень

Функция: `experiments/run_exp.py -> main()`

### 3.1 Фильтр групп задач
Для каждого `task_groups[group_name]`:
- если `enabled=false` — группа пропускается

### 3.2 Задача внутри группы
Далее:
1. `group_rows = [r for r in records if r["task_name"] in task_names]`
2. Для каждой `task_name`:
   - `task_type = rows[0]["task_type"]`
   - `texts = [r["text"] for r in rows]`
   - `labels = [r["label"] for r in rows]`

Условия, которые важно понимать:
- **Pair-задачи** (`semantic_similarity`, `paraphrase`) сейчас берут только `text`, а `text_pair` игнорируется в probing.
- Для `emotion` multi-label probing сейчас использует `label` (первый label как `primary_label`), а не весь список `labels`.

### 3.3 Модели
Для каждой включённой модели из `cfg["models"]`:
- `load_language_model(family)` загружает `model, tokenizer, device`

### 3.4 Уровни анализа
Для каждого `level` из `cfg["levels"]` вызывается:
`run_embedding_pipeline(..., level=level, ...)`

Путь сохранения результатов формируется как:
`artifacts/<task_group>/<task_name>/<family>/<level>/`

---

## 4) Пайплайн извлечения представлений + probing + декомпозиция + интервенции

Функция: `src/embeddings/pipeline.py -> run_embedding_pipeline(...)`

### 4.1 Точка входа: токенизация и forward
1. Создаётся `EmbeddingExtractor(model, tokenizer, max_length=max_length)`
2. `embedding_output = extractor.encode(texts, batch_size=batch_size)`

`EmbeddingExtractor.encode` делает:
- токенизацию батчами
- `padding="max_length"`, `truncation=True`, `max_length=self.max_length`
- forward: `model(**inputs, output_hidden_states=True)` (hidden states по всем слоям)
- возврат:
  - `hidden_states`: кортеж тензоров по слоям
  - `attention_mask`: общий mask по всей выборке

В логе виден прогресс по батчам.

### 4.2 Агрегация по уровню
`extract_all_layers` проходит по каждому `layer_tensor` и вызывает:
`aggregate_layer(hidden=layer_tensor, mask=attention_mask, level=level, ...)`

Результат по уровню:
- `text`:
  - `mean` pooling по `attention_mask` (или `cls`, если выбрано)
  - на выходе: тензор `[N, D]`
- `sentence`:
  - эвристика деления по пунктуации
  - соотнесение с токенами (без учёта точных границ субтокенов; приближённо)
  - на выходе: `list[Tensor]`, где каждый Tensor имеет вид `[num_sentences_i, D]`
- `token`:
  - берёт токены до `seq_len = sum(attention_mask)`
  - на выходе: `list[Tensor]` вида `[seq_len_i, D]`

Дальше для probing/decomposition представления приводятся к матрице `[N, D]`:
- токены/предложения в пределах примера усредняются (логика `pipeline._layer_to_sample_matrix`).

### 4.3 Сохранение эмбеддингов по слоям
`save_layer_embeddings` сохраняет для каждого слоя:
- `.../<level>_<text_strategy>_layer_<layer_idx>.npy`

Важно:
- для `token/sentence` файлы — это `dtype=object` (т.к. внутри различная длина списка токенов/предложений)

### 4.4 Probing
1. Для каждого слоя обучается линейный зонд:
   - classification -> `StandardScaler + LogisticRegression`
   - regression -> `StandardScaler + Ridge`
2. Разбиение train/test:
   - `train_test_split(..., test_size=0.2, random_state=seed, stratify=...)`

Результат probing:
- словарь `{"0": score, "1": score, ...}` (score зависит от метрики task_type)

### 4.5 Декомпозиция (улучшенная)
Декомпозиция считается на **последнем слое** (`layer_outputs[-1]`) в матрице `[N, D]`.

Поддерживаемые методы (из конфига):
1. `pca`:
   - PCA на `[N, D]`
   - сохраняются:
     - `components`
     - `explained_variance_ratio`
     - `projections` (внутренне)
2. `probe_directions`:
   - обучается линейный probe на `(x, y)`
   - направления берутся из `coef_`, пересчитываются назад из scaled-space в исходное пространство признаков
3. `null_space`:
   - прямой “null space” как удаление компоненты реализован позже в интервенциях

### 4.6 Интервенционная проверка (удаление компонент)
Для каждого метода из `decomposition.interventions` выполняется:

1. `pca`:
   - проекция-вычитание на первые `drop_components` PCA directions
   - повторный training probing
2. `probe_directions`:
   - проекция-вычитание на первые `drop_components` probe directions
   - повторный training probing
3. `null_space`:
   - “удаление” через ортогональное проектирование в комплемент span(directions)
   - повторный training probing

Замечание по форме результатов:
- при интервенции передаётся только одна матрица признаков как “один слой”, поэтому ответы probing для интервенции обычно имеют ключ `layer 0`.

### 4.7 Итоговые артефакты
В `pipeline_summary.json` сохраняется:
- `embedding_paths`
- `probing` (по слоям без интервенций)
- `decomposition` (кратко, без полной матрицы компонент)
- `intervention` (результаты probing после каждой интервенции)

---

## 5) Что важно знать (ограничения текущего кода)

1. Pair-задачи (`semantic_similarity`, `paraphrase`):
   - используется только `record["text"]`
   - `record["text_pair"]` игнорируется
2. Emotion (GoEmotions multi-label):
   - `record["labels"]` сохраняется, но probing сейчас идёт по `record["label"]` (primary label)
3. Для `token/sentence`:
   - декомпозиция и probing считаются после усреднения токенов/предложений в пределах одного примера (чтобы получить матрицу `[N, D]`).

---

## 6) Типичный порядок событий (коротко)

1. `run_exp.main()`
2. `setup_experiment_logging()`
3. `load_all_datasets()`
4. for `task_group` (enabled)  
5. for `task_name`  
6. for `family` (enabled)  
7. for `level` in cfg["levels"]  
8. `run_embedding_pipeline()`:
   - `extractor.encode()` (hidden states по слоям)
   - `aggregate_layer()` (token/sentence/text)
   - `save_layer_embeddings()`
   - `train_probes_by_layer()` (probing)
   - `decompose_embeddings()` (PCA + probe directions)
   - `intervention_with_decomposition()` (pca/probe/null_space)
   - запись `pipeline_summary.json`

