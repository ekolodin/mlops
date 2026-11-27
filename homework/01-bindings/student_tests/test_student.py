"""Свои тесты. Запускаются против установленного wheel, ожидаемые ответы задаются независимо от mac."""
import numpy as np
import pytest

from tensor_ops import mac


def test_elementwise_not_matmul():
    # Числа посчитаны вручную. При матричном умножении было бы [[19.5, 22.5], [43.5, 50.5]].
    a = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    b = np.array([[[5.0, 6.0], [7.0, 8.0]]])
    c = np.full((1, 2, 2), 0.5)
    np.testing.assert_allclose(mac(a, b, c), [[[5.5, 12.5], [21.5, 32.5]]], rtol=1e-12, atol=1e-12)


def test_copied_float64_dtype_is_accepted():
    # Тот же float64, но другой объект dtype: проверка через is вместо equal его бы отклонила.
    dtype = np.dtype("float64", copy=True)
    a = np.arange(24.0).reshape(2, 3, 4).view(dtype)
    assert a.dtype is not np.dtype("float64")
    b = np.full((2, 3, 4), 2.0)
    c = np.ones((2, 3, 4))
    np.testing.assert_allclose(mac(a, b, c), np.arange(24.0).reshape(2, 3, 4) * 2.0 + 1.0,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("make_bad", [
    lambda: np.asfortranarray(np.ones((2, 3, 4))),
    lambda: np.ones((4, 3, 2)).transpose(2, 1, 0),
])
def test_fortran_and_transposed_layout_rejected(make_bad):
    bad = make_bad()
    assert bad.shape == (2, 3, 4) and not bad.flags.c_contiguous
    good = np.ones((2, 3, 4))
    with pytest.raises(ValueError):
        mac(bad, good, good)


def test_scalar_and_zero_dim_are_rejected():
    good = np.ones((1, 1, 1))
    with pytest.raises(TypeError):
        mac(1.0, good, good)
    with pytest.raises(ValueError):
        mac(np.array(1.0), good, good)


def test_same_object_for_all_inputs():
    x = np.array([[[1.0, -2.0, 3.0]]])
    out = mac(x, x, x)
    np.testing.assert_allclose(out, [[[2.0, 2.0, 12.0]]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, [[[1.0, -2.0, 3.0]]])
    assert not np.shares_memory(out, x)


def test_contiguous_view_with_offset():
    # Срез по первой оси C-contiguous, но начинается не с начала буфера.
    big = np.arange(48.0).reshape(4, 3, 4)
    view = big[1:3]
    assert view.flags.c_contiguous and not view.flags.owndata
    ones = np.ones((2, 3, 4))
    expected = np.arange(12.0, 36.0).reshape(2, 3, 4) + 1.0
    np.testing.assert_allclose(mac(view, ones, ones), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(big, np.arange(48.0).reshape(4, 3, 4))


def test_rounding_matches_numpy_on_cancellation():
    # x * x не представимо точно. NumPy округляет произведение, и fl(x*x) + (-fl(x*x)) = 0.
    # Если компилятор сольёт a * b + c в FMA, вместо нуля будет ошибка округления до 1e-8.
    x = np.array([[[1000.1, 3.3, 7.7, 12345.678]]])
    c = -(x * x)
    np.testing.assert_allclose(mac(x, x, c), np.zeros((1, 1, 4)), rtol=1e-12, atol=1e-12)
