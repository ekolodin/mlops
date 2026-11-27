# Версии данных: Titanic, CatBoost и DVC

Обучим CatBoost на двух версиях Titanic и сохраним данные через DVC.
Затем вернём первую версию и проверим, почему одного Git-коммита для этого недостаточно.

Нужны Python 3.13, uv и Git. Команды рассчитаны на bash/zsh в macOS или Linux.

## Основные команды DVC

Кэш — локальное хранилище содержимого файлов. Remote — настроенное хранилище,
куда отправляем данные и откуда их получаем.

| Пример команды | Что делает |
|---|---|
| `dvc init` | Создаёт настройки DVC в текущем Git-репозитории. |
| `dvc remote add -d storage remote-storage` | Подключает папку `remote-storage` под именем `storage`. Флаг `-d` выбирает её по умолчанию. |
| `dvc remote list` | Показывает имена и адреса подключённых хранилищ. |
| `dvc add data/train-split.csv` | Сохраняет содержимое CSV в кэш, создаёт или обновляет указатель `.dvc`, исключает CSV из Git. В remote ничего не отправляет. |
| `dvc status` | Проверяет, соответствуют ли данные текущим указателям и доступны ли нужные объекты в кэше. |
| `dvc push` | Загружает из кэша в выбранный remote недостающие объекты текущей версии. Git-коммит не создаёт. |
| `dvc fetch` | Скачивает нужные объекты из remote в кэш, не меняя рабочие CSV. |
| `dvc pull` | Скачивает недостающие объекты в кэш и восстанавливает рабочие файлы по текущим указателям. |
| `dvc checkout` | Восстанавливает рабочие файлы из кэша по текущим указателям, например после переключения Git-коммита. Из remote ничего не скачивает. |
| `dvc diff data-v1 data-v2` | Сравнивает версии: показывает добавленные, удалённые и изменённые файлы, но не построчные изменения CSV. |
| `dvc add --help` | Показывает справку и параметры команды `add`. |

`dvc pull` выполняет загрузку и восстановление: по смыслу это `dvc fetch` + `dvc checkout`.
Версию указателей выбираем через Git; сами данные восстанавливаем через DVC.
Подробнее — в [справочнике команд DVC](https://doc.dvc.org/command-reference).

## Подготовка

Откройте `03-data-storage/practice` в клоне репозитория курса:

```bash
uv sync --locked
source .venv/bin/activate
python --version
dvc --version
python -m pytest -q
```

Ожидаем пять пройденных тестов. Env находится в `.venv`, версии пакетов — в `uv.lock`.
Дальнейшие команды выполняйте в этом же терминале с активированным env.

Создайте Git-репозиторий в `work`, чтобы переключать версии отдельно от репозитория курса:

```bash
mkdir work
cp prepare_data.py train.py infer.py .gitignore work/
cd work
git init -b main
git add prepare_data.py train.py infer.py .gitignore
git commit -m "Add Titanic scripts"
```

## 1. Подготовить две выборки

`Survived`: 1 — пассажир выжил, 0 — погиб. В Kaggle `test.csv` нет ответов,
поэтому собственный test выделяем из размеченного `train.csv`.

```bash
python prepare_data.py
wc -l data/train-split.csv data/test-split.csv
mkdir backup
cp data/train-split.csv backup/train-split.csv
```

Скрипт скачивает [архив, используемый CatBoost](https://github.com/catboost/catboost/blob/master/catboost/python-package/catboost/datasets.py),
проверяет контрольную сумму и делит 891 запись: 713 в train, 178 в test.
Seed фиксирован. `wc -l` покажет 714 и 179 строк с учётом заголовков.
Исходный CSV остаётся в `data/raw/train.csv`.

Для скачанного CSV: `python prepare_data.py --input /путь/к/train.csv`.
Это альтернатива первой команде; существующие сплиты скрипт не перезаписывает.

Оставьте в обучающем CSV первые 249 объектов. В файле будет 250 строк с заголовком:

```bash
python -c 'import pandas as pd; p="data/train-split.csv"; pd.read_csv(p).head(249).to_csv(p, index=False)'
wc -l data/train-split.csv
```

Полная обучающая выборка сохранена в `backup`. Тестовую выборку больше не меняем.

## 2. Обучить и проверить модель

В `train.py` валидация выделяется из train. Внешний `test-split.csv`
при обучении не читается.

```bash
python train.py --train data/train-split.csv --save-path artifacts/model-v1.cbm
python infer.py --model-path artifacts/model-v1.cbm --test data/test-split.csv
```

Первая команда выводит ROC-AUC на валидации, вторая — на тесте.
ROC-AUC оценивает, насколько модель ставит выживших выше погибших по вероятности:
0.5 — случайное ранжирование, 1 — идеальное.
Запишите `test_roc_auc` для v1.

## 3. Сохранить первую версию данных

```bash
dvc init
dvc add data/train-split.csv data/test-split.csv
cat data/train-split.csv.dvc
cat data/.gitignore
git status --short
```

Файл-указатель `.dvc` содержит путь, размер и хеш — отпечаток содержимого CSV.
Сам CSV исключён из Git, а его содержимое сохранено в кэш `.dvc/cache`.
Во внешнее хранилище данные пока не отправлены.

Настройте remote — хранилище данных. Здесь используем локальную папку:

```bash
mkdir remote-storage
dvc remote add -d storage remote-storage
cat .dvc/config
dvc push
find remote-storage -type f
git add .dvc .dvcignore data/.gitignore data/train-split.csv.dvc data/test-split.csv.dvc
git commit -m "Track small training dataset"
git tag data-v1
```

В remote появились два объекта. Сравните их пути с `.dvc/cache/files/md5`:
имена связаны с хешами. Remote исключён из Git.
Вместо папки можно подключить S3 или SSH; отдельный сервер DVC не нужен.

Тег `data-v1` — имя Git-коммита, по которому вернём первую версию.

Что потеряется, если сохранить Git-репозиторий, но не сохранить CSV, кэш и remote?

## 4. Добавить данные и сохранить вторую версию

```bash
cp backup/train-split.csv data/train-split.csv
wc -l data/train-split.csv
python train.py --train data/train-split.csv --save-path artifacts/model-v2.cbm
python infer.py --model-path artifacts/model-v2.cbm --test data/test-split.csv
dvc status
```

Train снова содержит 713 объектов. Сравните `test_roc_auc` с v1.
Увеличение выборки не гарантирует улучшения метрики.

```bash
dvc add data/train-split.csv
git diff -- data/train-split.csv.dvc
dvc push
git add data/train-split.csv.dvc
git commit -m "Use full training dataset"
git tag data-v2
find remote-storage -type f
```

Изменились хеш и размер указателя train. В remote теперь три объекта:
две полные версии train и одна версия test. DVC не сохраняет построчный diff CSV.

## 5. Восстановить первую версию

`--detach` открывает коммит вне ветки. Здесь новых коммитов не создаём.
Предположите, сколько строк покажет `wc -l` после переключения:

```bash
git switch --detach data-v1
cat data/train-split.csv.dvc
wc -l data/train-split.csv
dvc status
```

Git вернул старый указатель, но CSV пока содержит 714 строк. Восстановите данные:

```bash
dvc checkout
wc -l data/train-split.csv
dvc status
```

Теперь 250 строк: `dvc checkout` восстановил версию из кэша.
Если в кэше её нет, но она есть в remote, используйте `dvc pull`.

Вернитесь к полной выборке самостоятельно. Сначала выберите в Git ветку `main`,
затем восстановите соответствующие ей данные. Проверьте 714 строк и чистый статус DVC.

DVC отслеживает только два CSV; модели в `artifacts` при откате не меняются.
Что нужно добавить, чтобы восстанавливать вместе с данными модель (`.cbm`)?
