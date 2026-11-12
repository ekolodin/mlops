# ДЗ-1. Python-пакет с C++-биндингами

Гостева Марина, М13-601.

## Решение

`tensor_ops.mac(a, b, c)` вычисляет поэлементное выражение `a * b + c`.
В `kernel.cpp` используется один цикл по элементам. Входные указатели —
`const double*`, запись идёт в отдельный выходной массив.

В `bindings.cpp` проверяются dtype, размерность, формы и расположение данных
в памяти. Dtype сравнивается через `.equal()`, чтобы принимать разные объекты,
описывающие один и тот же тип. `.noconvert()` запрещает неявные преобразования.
После проверок создаётся `py::array_t<double>` и вызывается ядро.
При нулевом размере возвращается пустой результат без обращения к буферам.

Контракт функции:

- Три `numpy.ndarray` одинаковой 3D-формы, native-endian `float64`.
- Входы C-contiguous и aligned; read-only массивы допустимы.
- Результат имеет ту же форму и собственный буфер; входы не изменяются.
- Неверный dtype или тип объекта — `TypeError`; форма, ndim или layout — `ValueError`.
- Broadcasting и автоматическое преобразование входов не поддерживаются.

CMake собирает `_core` из двух C++-файлов с C++17 и устанавливает его
в `tensor_ops`. Сборкой управляет `scikit-build-core`.
Build dependencies: `scikit-build-core`, `pybind11`, CMake, Ninja и компилятор.
`build` запускает сборку, `pytest` нужен для тестов.
Runtime dependency — `numpy==2.5.3`; она указана в metadata wheel.
Установке готового wheel инструменты сборки не нужны.

## Воспроизведение из чистого клона

Команды рассчитаны на bash: Linux, WSL или macOS с подходящим компилятором.
Нужны Python 3.13 и доступ к индексам Python-пакетов. Выполнять из корня клона
с веткой решения; `starter/dist` до сборки должен быть пустым или отсутствовать.

```bash
cd homework/01-bindings
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check

python -m build example --outdir example/dist
python -m pip install example/dist/*.whl
python -c 'from binding_demo import add; print(add(2.0, -3.5))'
python -m pytest example/tests -q

python -m build starter --outdir starter/dist
python -m pip install --force-reinstall starter/dist/*.whl
python -m pip check
python -m pytest tests student_tests -q
```

`python -m build` сначала создаёт sdist, а затем собирает из него wheel.
В проверенном прогоне обе стадии завершились успешно. Пример вернул `-1.5`,
его тест завершился с результатом `1 passed`.

Проверка установки в новый env, вне каталога исходников:

```bash
hw_root="$PWD"
hw_receiver=$(mktemp -d)
python -m venv "$hw_receiver/venv"
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip list
env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip install \
  pytest==9.1.1 "$hw_root"/starter/dist/*.whl
cp -R tests student_tests "$hw_receiver/"
(
  cd "$hw_receiver"
  env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip check
  env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pytest tests student_tests -q
  env -u PYTHONPATH "$hw_receiver/venv/bin/python" -c \
    'import tensor_ops._core as m; print(m.__file__)'
  env -u PYTHONPATH "$hw_receiver/venv/bin/python" -m pip list
)
python -c 'from pathlib import Path; import hashlib; ws = list(Path("starter/dist").glob("*.whl")); assert len(ws) == 1; p = ws[0]; print(p.name); print(hashlib.sha256(p.read_bytes()).hexdigest())'
```

В новом env до установки был только `pip`. NumPy отдельно не устанавливался:
он был получен из зависимости wheel. `requirements-dev.txt` в env получателя
не устанавливался; `--no-deps`, editable-установка, `--system-site-packages`
и подмена пути импорта не использовались. В папку получателя скопированы только
тесты, исходников пакета там нет.

## Фактические результаты проверки

Дата: 6 ноября 2026 года. База решения — коммит курса
`1720bcc6842a7ec6f32555db41dc589452e15ec8`.

Контрольный прогон выполнен на Mac с помощью `bash check_local.sh`.
Скрипт собрал пример и пакет, затем установил wheel в отдельный чистый env.
Каталог прогона: `/Users/marina/Downloads/mlops-hw01/local-check.AmtXdy`.
Команды выше описывают тот же порядок ручной проверки из чистого клона;
в скрипте build-env, dist и receiver размещены внутри каталога прогона.

ОС: **macOS 26.5.1 (25F80)**, архитектура **arm64 (Apple Silicon)**. Компилятор:
`Apple clang 21.0.0 (clang-2100.1.1.101)`.

Основные команды фактического прогона (пути сокращены через переменные):

```bash
hw="/Users/marina/Downloads/mlops-hw01/homework/01-bindings"
work="/Users/marina/Downloads/mlops-hw01/local-check.AmtXdy"
unset PYTHONPATH
"$work/build-env/bin/python" -m build "$hw/starter" --outdir "$work/dist"
python3.13 -m venv "$work/receiver/venv"
"$work/receiver/venv/bin/python" -m pip install pytest==9.1.1 "$work/dist/mlops_tensor_ops-0.1.0-cp313-cp313-macosx_26_0_arm64.whl"
cp -R "$hw/tests" "$hw/student_tests" "$work/receiver/"
cd "$work/receiver"
"$work/receiver/venv/bin/python" -m pip check
"$work/receiver/venv/bin/python" -m pytest tests student_tests -q
"$work/receiver/venv/bin/python" -c 'import tensor_ops._core as m; print(m.__file__)'
"$work/receiver/venv/bin/python" -m pip list
```

Версии Python и инструментов:

```text
3.13.16
build==1.6.1
pybind11==3.1.0
scikit-build-core==1.0.3
cmake==4.4.3
ninja==1.13.2
numpy==2.5.3
pytest==9.1.1
```

Проверенный wheel:

```text
mlops_tensor_ops-0.1.0-cp313-cp313-macosx_26_0_arm64.whl
SHA-256: abd12dfb217beb1dd6db5c30b9c4125a8989f9ef1711edd3d819e2d03847833e
```

Хеш относится к этому конкретному файлу. Повторная сборка может создать файл
с другим хешем; побитовая воспроизводимость wheel не заявляется.

Тесты в чистом env:

```text
...............................................................          [100%]
63 passed in 4.56s
```

Это 40 выданных и 23 собственных проверки с учётом параметризации.
Вывод `pip check`:

```text
No broken requirements found.
```

Фактический путь `_core`:

```text
/Users/marina/Downloads/mlops-hw01/local-check.AmtXdy/receiver/venv/lib/python3.13/site-packages/tensor_ops/_core.cpython-313-darwin.so
```

Список пакетов env получателя:

```text
Package          Version
---------------- -------
iniconfig        2.3.1
mlops-tensor-ops 0.1.0
numpy            2.5.3
packaging        26.3
pip              26.2.1
pluggy           1.7.0
Pygments         2.21.0
pytest           9.1.1
```

В этом списке нет `build`, `scikit-build-core`, `pybind11`, `cmake` и `ninja`.
Расширение загружено из `site-packages` нового env, что дополнительно проверяет
выданный тест `test_distribution.py`.

## Свои тесты и найденная ошибка

В `student_tests/test_mac_extra.py` добавлены семь тестовых функций:

1. Дробные и отрицательные числа, форма `(2, 1, 3)`, ответ рассчитан вручную.
2. Fortran-layout, перестановка осей и отрицательный stride в каждом аргументе.
3. Один и тот же read-only массив во всех трёх аргументах; независимость результата.
4. Сохранность результата после удаления входов.
5. Допустимый dtype, равный `float64`, но созданный как отдельный объект.
6. Проверка неверного dtype, числа осей и формы даже на пустых массивах.
7. Пустая форма `(0, 0, 0)` и собственный буфер результата.

При предварительной проверке на Linux проверена чувствительность численного теста к ошибке. В отдельной временной
копии ядра строка `out[i] = a[i] * b[i] + c[i]` заменена на
`out[i] = a[i] * b[i]`: потеряно прибавление третьего массива. Из этой копии
собран и установлен отдельный wheel в env разработчика, после чего выполнено:

```bash
python -m pytest student_tests/test_mac_extra.py -k fractional_values -q
```

Тест упал: `1 failed, 22 deselected`. Отличались все 6 элементов, максимальная
абсолютная ошибка составила 7. Например, для первого элемента получено 2 вместо 3.
Дефект внесён специально для проверки теста.

Затем установлен исходный исправный wheel; тот же тест дал
`1 passed, 22 deselected`. Последующий полный прогон в новом env дал 63 успешных
проверки. В итоговых исходниках дефекта нет.

## Ограничения и источники

Реализация намеренно ограничена контрактом задания: только три 3D-массива
`float64`, без broadcasting и поддержки strided-входов. Ускорение относительно
NumPy не измерялось и не заявляется. Специальные гарантии для NaN/inf,
переполнения и параллельного изменения входов не добавлялись.

Проверен CPython 3.13 на macOS arm64 и ранее на Linux x86_64.
Нативная сборка на Windows не проверялась. Wheel имеет платформенный тег и не подходит для произвольной
ОС, архитектуры или версии Python. Флаги `-march=native` и `-ffast-math`
в проект не добавлены.

Использованы материалы репозитория курса:

- [Условие и подсказки по API](https://github.com/ekolodin/mlops/blob/1720bcc6842a7ec6f32555db41dc589452e15ec8/homework/01-bindings/README.md).
- [Рабочий пример сборки](https://github.com/ekolodin/mlops/tree/1720bcc6842a7ec6f32555db41dc589452e15ec8/homework/01-bindings/example).
- [Выданные тесты контракта](https://github.com/ekolodin/mlops/tree/1720bcc6842a7ec6f32555db41dc589452e15ec8/homework/01-bindings/tests).

Выданные `tests/`, учебный `example/` и материалы других тем не изменены.
