# Практика 02. Качество кода и документация

Продолжаем taxi-проект: приводим код к общим правилам, выделяем расчёт метрик,
документируем API и подключаем проверки перед коммитом. Обучение и предсказание
уже реализованы; поведение модели должно сохраниться.

## Подготовка

Из корня репозитория `mlops`:

```bash
cd 02-code-and-data/practice
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -c 'import taxi_duration; print(taxi_duration.__file__)'
python -m pytest -q
```

Команды приведены для bash/zsh. На Windows используйте Git Bash:
создайте env через `py -3.13 -m venv .venv`, активируйте через
`source .venv/Scripts/activate`; остальные команды те же.

Путь импорта должен вести в `02-code-and-data/practice/src/taxi_duration`.
Используйте отдельный env: пакет называется так же, как в первой практике.
Исходный результат — **22 passed**. Тесты выполняются на маленьких таблицах,
которые создаются внутри тестов.

Далее все Python-команды выполняются из `02-code-and-data/practice`
с активированным env. Блоки с `cd ../..` возвращают вас обратно автоматически.

## Что открыть

| Файл | Задача |
|---|---|
| `pyproject.toml` | TODO 1: правила Ruff |
| `src/taxi_duration/data.py`, `model.py`, `cli.py` | TODO 1: исправления стиля и замечаний линтера |
| `src/taxi_duration/metrics.py`, `model.py` | TODO 2: выделение расчёта RMSE |
| `tests/test_student.py` | TODO 2: собственные тесты метрики |
| `data.py`, `model.py`, `docs/` | TODO 3: аннотации, docstrings, Sphinx |
| `.pre-commit-config.yaml` | TODO 4: автоматический запуск Ruff |

## TODO 1. Настроить Ruff

Добавьте в `pyproject.toml` настройки:

- `target-version` — `py313`;
- `line-length` — `88`;
- в `[tool.ruff.lint]` включите `E4`, `E7`, `E9`, `F`, `I`, `B`.

Запустите проверку:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
```

Прочитайте сообщения. Найдите неиспользуемый импорт и локальную переменную,
затем исправьте их. Подсказка: `F401` относится к импорту, `F841` — к переменной.
Не подавляйте сообщения через `noqa` и не исключайте проблемные файлы из проверки.

Часть исправлений можно поручить инструменту:

```bash
python -m ruff check --fix src tests
python -m ruff format src tests
git diff -- src tests pyproject.toml
python -m pytest -q
```

`--fix` может завершиться с ошибкой, если остались замечания для ручного исправления.
Не включайте `--unsafe-fixes`. После исправлений повторите обе проверки:
линтер и `format --check` должны завершиться без ошибок.

## TODO 2. Выделить расчёт метрик

В `metrics.py` реализуйте `regression_metrics(y_true, y_pred)`:

- входы — непустые одномерные массивы или списки одинаковой длины;
- результат — `{"rmse": ...}`, значение — Python `float`;
- используйте `sklearn.metrics.root_mean_squared_error`;
- функция не меняет входы, не обучает модель и не читает файлы.

Сначала напишите в `tests/test_student.py` тесты для трёх примеров:

| Истинные значения | Предсказания | Что проверить |
|---|---|---|
| `[1, 3]` | `[2, 2]` | RMSE = 1.0 |
| `[2, 5]` | `[2, 5]` | RMSE = 0.0 |
| `[0, 0]` | `[0, 4]` | Посчитайте RMSE вручную; это не MSE |

Проверяйте словарь и значение через `pytest.approx`. Убедитесь, что тесты
не проходят без реализации, затем допишите функцию.

В `train_model` замените все три вычисления RMSE вызовами новой функции.
Сохраните:

- имена `train_rmse`, `validation_rmse`, `baseline_rmse`;
- train и validation на прежних местах;
- baseline по **среднему train**;
- остальные поля отчёта и логику обучения.

Удалите импорт, который больше не нужен в `model.py`, и запустите тесты.
С тремя новыми тестами получится **25 passed**. Больше проверок можно добавить
для пустых входов, несовпадающих длин и неизменности массивов.

## TODO 3. Описать API и собрать документацию

Добавьте аннотации и Google-style docstrings двум функциям:

**`prepare_training_data`** в `data.py`:

- точные имена четырёх обязательных колонок;
- новая колонка `duration` в минутах и фильтр `[1, 60]` включительно;
- исключение строк с некорректным временем;
- исходная таблица не меняется, результат может быть пустым;
- когда возникает `ValueError`.

**`predict`** в `model.py`:

- содержимое `bundle` и необходимые колонки запроса;
- target и времена поездки не нужны;
- результат — массив длительностей в минутах;
- число и порядок строк сохраняются, пустой вход допустим;
- исходная таблица не меняется, vectorizer не переобучается;
- когда возникает `ValueError`.

Используйте `pd.DataFrame`, `np.ndarray` и `dict` там, где они подходят.
Описывайте существующее поведение, не изменяйте реализацию ради документации.

В `docs/conf.py` подключите `sphinx.ext.autodoc` и `sphinx.ext.napoleon`.
В `docs/api.rst` добавьте обе функции через директивы `autofunction` или
`automodule` с `:members:`. Пример синтаксиса для одной функции:

```rst
.. autofunction:: taxi_duration.data.prepare_training_data
```

В `docs/index.rst` добавьте небольшой пример вызова подготовки данных.
Обратите внимание: отступы после `.. code-block:: python` значимы.

```bash
python -m sphinx -b html -W docs docs/_build/html
```

Откройте `docs/_build/html/index.html`, перейдите на страницу API и проверьте,
что видны **обе функции**, их аргументы, результат и ошибки. Успешная сборка
пустого каркаса ещё не означает, что API подключён. Измените одну фразу в
docstring, пересоберите документацию и найдите изменение в HTML.

## TODO 4. Подключить pre-commit

В `.pre-commit-config.yaml` уже есть рабочий пример проверки YAML.
Добавьте два hook в существующий `repo: local`:

| Поле | Линтер | Форматтер |
|---|---|---|
| `id` | `ruff-check` | `ruff-format` |
| `entry` | `python -m ruff check --fix` | `python -m ruff format` |
| `language` | `system` | `system` |
| `types` | `[python]` | `[python]` |

Линтер должен идти **перед** форматтером. У обоих задайте `files`:
`^02-code-and-data/practice/(src|tests)/.*\.py$`.
Не добавляйте `.` в `entry`: подходящие имена файлов передаёт pre-commit.

Глобальный `files` ограничивает hooks второй практикой. Пути считаются от
корня Git. `language: system` означает, что инструменты берутся из активного
env; отдельные окружения hooks здесь не создаются.

Установите конфигурацию и запустите проверки из корня репозитория:

```bash
(
  cd ../..
  python -m pre_commit install --config 02-code-and-data/practice/.pre-commit-config.yaml
  python -m pre_commit run --config 02-code-and-data/practice/.pre-commit-config.yaml --all-files
)
```

Новые файлы, например собственные тесты, сначала добавьте через `git add`:
`--all-files` проверяет отслеживаемые файлы, а не произвольные файлы на диске.

Проверьте настоящий коммит:

1. Внесите нарушение оформления в `data.py`, не меняя смысл кода.
2. Добавьте изменения через `git add` и попробуйте создать локальный коммит.
3. Hook исправит файл и остановит коммит. Прочитайте вывод и `git diff`.
4. Снова добавьте исправленный файл и повторите коммит.

Не обходите hooks через `--no-verify`. Для этого упражнения push не нужен.

## Итоговая проверка

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
python -m sphinx -b html -W docs docs/_build/html
```

Проверьте также, что `train_model` действительно использует новую функцию
для трёх метрик, в HTML есть оба описания API, а hooks останавливают коммит
при нарушении. Одних исходных 22 тестов для завершения всех TODO недостаточно.

Для дополнительного запуска на данных первой практики:

```bash
python -m taxi_duration.cli train \
  --train ../../01-mlops-fundamentals/practice/data/train.parquet \
  --validation ../../01-mlops-fundamentals/practice/data/validation.parquet \
  --output artifacts
```

Повторный запуск требует нового имени папки результата: CLI не перезаписывает
готовые артефакты. Тесты и обязательные задания не требуют запуска на этих файлах.
