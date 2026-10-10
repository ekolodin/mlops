import numpy as np
import pytest

from tensor_ops import mac


def test_manual_result():
    """Численный ответ задан вручную."""
    a = np.array([1, -2, 0, 3], dtype=np.float64).reshape(1, 2, 2)
    b = np.array([4, 5, -7, -2], dtype=np.float64).reshape(1, 2, 2)
    c = np.array([0.5, 1, 2, 3], dtype=np.float64).reshape(1, 2, 2)
    expected = np.array([4.5, -9, 2, -3]).reshape(1, 2, 2)

    result = mac(a, b, c)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_fortran_layout_in_second_argument():
    """Второй аргумент с Fortran-layout должен быть отклонён."""
    a = np.ones((2, 3, 4), dtype=np.float64)
    b = np.asfortranarray(a)
    c = np.zeros_like(a)

    assert not b.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(a, b, c)


def test_same_readonly_input_and_independent_output():
    """Один read-only массив можно передать трижды."""
    x = np.array([1, 2, 3, 4], dtype=np.float64).reshape(1, 2, 2)
    original = x.copy()
    x.setflags(write=False)
    expected = np.array([2, 6, 12, 20], dtype=np.float64).reshape(1, 2, 2)

    result = mac(x, x, x)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert not np.shares_memory(result, x)

    result.fill(-100)
    np.testing.assert_array_equal(x, original)