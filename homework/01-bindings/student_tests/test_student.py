"""Student tests for tensor_ops.mac(a, b, c).

Проверяют три вещи, независимо от выданных tests/test_contract.py:
1. Численный результат на вручную посчитанном примере.
2. Негативный сценарий: Fortran-layout отклоняется (не silent copy).
3. Граница/память: входы-псевдонимы и read-only входы не изменяются,
   а выход не делит память с входами.
"""

import numpy as np
import pytest

from tensor_ops import mac


def test_hand_calculated_3d_values():
    """Один элемент каждой позиции — считаем ответ вручную."""
    a = np.array([[[1.0, 2.0],
                   [3.0, 4.0]],
                  [[5.0, 6.0],
                   [7.0, 8.0]]], dtype=np.float64)
    b = np.array([[[10.0, 20.0],
                   [30.0, 40.0]],
                  [[50.0, 60.0],
                   [70.0, 80.0]]], dtype=np.float64)
    c = np.array([[[0.5, 0.5],
                   [0.5, 0.5]],
                  [[0.5, 0.5],
                   [0.5, 0.5]]], dtype=np.float64)
    # Ожидание: a*b + c, посчитанное вручную построчно.
    expected = np.array([[[10.5, 40.5],
                          [90.5, 160.5]],
                         [[250.5, 360.5],
                          [490.5, 640.5]]], dtype=np.float64)
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_fortran_layout_is_rejected():
    """Fortran-порядок памяти не C-contiguous, должен дать ValueError.

    Если бы реализация молча делала copy, контракт "inputs must be C-contiguous"
    был бы нарушен, и вызывающий код не узнал бы о проблеме.
    """
    a = np.asfortranarray(np.ones((2, 3, 4), dtype=np.float64))
    b = np.ones((2, 3, 4), dtype=np.float64)
    c = np.ones((2, 3, 4), dtype=np.float64)
    assert not a.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(a, b, c)


def test_readonly_and_aliasing_inputs_are_safe():
    """Read-only входы допустимы; выход не делит память с входами.

    Здесь же ловим ошибку "выход — это view на один из входов":
    если бы реализация вернула тот же буфер, out.fill(...) испортил бы вход.
    """
    a = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    b = np.full((2, 3, 4), 2.0, dtype=np.float64)
    c = np.full((2, 3, 4), 1.0, dtype=np.float64)
    expected = a * b + c

    for arr in (a, b, c):
        arr.flags.writeable = False

    out = mac(a, b, c)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    for original in (a, b, c):
        assert not np.shares_memory(original, out)

    out.fill(-123.0)
    np.testing.assert_allclose(a, np.arange(24.0).reshape(2, 3, 4),
                               rtol=1e-12, atol=1e-12)