# ДЗ-1: Python-пакет с C++-биндингами

Шибаев Александр, М05-611.

## Решение

`mac(a, b, c)` считает `a * b + c` поэлементно. В `kernel.cpp` обычный цикл
по элементам, в `bindings.cpp` — проверки входов и выделение результата.

Входы должны быть трёхмерными NumPy-массивами одинаковой формы, с native-endian
`float64`, флагами `C_CONTIGUOUS` и `ALIGNED`. Неверный тип даёт `TypeError`,
неверная форма или расположение данных — `ValueError`. Автоматическое
преобразование и broadcasting запрещены.

Read-only входы и пустые массивы допустимы. Входы не меняются, результат
владеет отдельным буфером. Для пустых массивов проверки остаются, а вызов
ядра пропускается.

Для сборки нужны компилятор C++17, scikit-build-core, pybind11, CMake и Ninja.
Для использования готового пакета нужен NumPy: `numpy==2.5.3` указан
в runtime-зависимостях. pytest нужен для тестов.

## Среда

Сборка и тесты: 30 сентября 2026 года, macOS arm64.

| Компонент | Версия |
|---|---|
| ОС | macOS 15.7.7, build 24G720 |
| Архитектура | arm64, Darwin |
| Python | CPython 3.13.9 |
| Компилятор | Apple clang 17.0.0 (clang-1700.4.4.1) |
| build | 1.6.1 |
| pybind11 | 3.1.0 |
| scikit-build-core | 1.0.3 |
| CMake | 4.4.3 |
| Ninja | 1.13.2 |
| NumPy | 2.5.3 |
| pytest | 9.1.1 |

Исходная версия курса: `51f9c918147b159ab42d05011ed1e7f27db7e705`.
Работа выполнена в ветке `hw01`; исходные `tests/`, `example/`
и файлы других тем не изменены.

## Сборка

В `dist` должен быть только один wheel.

```bash
git clone --branch hw01 https://github.com/Shibaev-am/mlops.git
cd mlops
python3.13 -m venv .venv
source .venv/bin/activate
cd homework/01-bindings
python -m pip install -r requirements-dev.txt
python -m pip check

python -m build example --outdir example/dist
python -m pip install example/dist/*.whl
python -c 'from binding_demo import add; print(add(2.0, -3.5))'
python -m pytest example/tests -q

python -m build starter --outdir starter/dist
python -m pip install starter/dist/*.whl
python -m pip check
python -c 'import tensor_ops; print(tensor_ops.__file__)'
python -m pytest tests student_tests -q
```

`python -m build` сначала собирает sdist, затем wheel из него.

Результаты:

```text
No broken requirements found.
-1.5
1 passed in 0.01s
Successfully built mlops_tensor_ops-0.1.0.tar.gz and mlops_tensor_ops-0.1.0-cp313-cp313-macosx_15_0_arm64.whl
68 passed in 1.47s
```

## Проверка wheel в чистом окружении

Команды из `homework/01-bindings` после сборки:

```bash
hw_root="$PWD"
hw_receiver=$(mktemp -d)
python3.13 -m venv "$hw_receiver/venv"
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
python - <<'PY'
from pathlib import Path
import hashlib
wheel, = Path("starter/dist").glob("*.whl")
print(wheel.name)
print(hashlib.sha256(wheel.read_bytes()).hexdigest())
PY
```

В новый venv установлены только wheel и pytest. Тесты скопированы отдельно
и запущены вне исходников, с удалённым `PYTHONPATH`. NumPy установился сам:

```text
Collecting numpy==2.5.3 (from mlops-tensor-ops==0.1.0)
Successfully installed iniconfig-2.3.0 mlops-tensor-ops-0.1.0 numpy-2.5.3 packaging-26.3 pluggy-1.6.0 pygments-2.21.0 pytest-9.1.1
```

`pip check`, тесты и путь `_core`:

```text
No broken requirements found.
....................................................................     [100%]
68 passed in 1.77s
/var/folders/rf/zm1yt1jn0ms0fk04s9lhs5s91vyh7v/T/hw01-receiver-14gpj4dv/venv/lib/python3.13/site-packages/tensor_ops/_core.cpython-313-darwin.so
```

`pip list` в окружении получателя:

```text
Package          Version
---------------- -------
iniconfig        2.3.0
mlops-tensor-ops 0.1.0
numpy            2.5.3
packaging        26.3
pip              25.2
pluggy           1.6.0
Pygments         2.21.0
pytest           9.1.1
```

В окружении нет build, scikit-build-core, pybind11, cmake и ninja.
Установка готового wheel не запускала компилятор.

Проверенный файл:

```text
mlops_tensor_ops-0.1.0-cp313-cp313-macosx_15_0_arm64.whl
SHA-256: ed67a1a4ee0dfa3ed5323e1a9475e2f88db7459df47f9c1532d25db8abf98653
```

Это хеш проверенного wheel; при пересборке он может измениться.

## Тесты

В `student_tests/test_mac_extra.py` семь тестов, 28 случаев с параметризацией:
дробные значения, один read-only массив во всех аргументах, отдельный объект
dtype, неправильный layout, пустые входы с ошибками, время жизни результата
и округление при компенсации произведения.
Вместе с выданными тестами — 68 проверок.

Тест `test_product_cancellation` воспроизвёл расхождение с NumPy.
При `a = b = 200.1` и `c = -(a * b)` NumPy возвращает ноль,
а первая сборка возвращала `2.96381585940253e-12` во всех 24 элементах.
Это больше допуска `atol=1e-12`:

```text
FAILED test_mac_extra.py::test_product_cancellation
1 failed, 27 deselected in 0.10s
```

Компилятор объединял умножение и сложение в FMA, где результат округляется
один раз. NumPy округляет произведение отдельно. В CMake для GCC/Clang
добавлен `-ffp-contract=off`, для MSVC — `/fp:strict`. После пересборки
проверка проходит вместе с остальными тестами.

## Источники и ограничения

Источники: [условие и подсказки по pybind11](https://github.com/ekolodin/mlops/tree/main/homework/01-bindings),
[пример сборки из курса](https://github.com/ekolodin/mlops/tree/main/homework/01-bindings/example)
и [документация Clang по floating-point contraction](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffp-contract).

Скорость с NumPy не сравнивалась. GIL не освобождается. NaN/inf и переполнение
не проверялись, по условию они не обязательны. Флаг `-march=native` не используется.
Wheel собран для CPython 3.13, macOS 15+ и arm64. Для других платформ нужна
пересборка; Linux и Windows не проверялись.
