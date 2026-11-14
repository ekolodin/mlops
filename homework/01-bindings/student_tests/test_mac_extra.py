import numpy as np
import pytest

from tensor_ops import mac


def test_known_values_3x1x2():
    a = np.array([[[2.0, 3.0]], [[-1.0, 0.5]], [[10.0, 0.0]]])
    b = np.array([[[4.0, -1.0]], [[-1.0, 8.0]], [[0.1, 7.0]]])
    c = np.array([[[1.0, 1.0]], [[0.0, -4.0]], [[-1.0, 2.5]]])
    expected = np.array([[[9.0, -2.0]], [[1.0, 0.0]], [[0.0, 2.5]]])
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_same_object_for_all_args():
    x = np.arange(1.0, 9.0).reshape(2, 2, 2)
    before = x.copy()
    out = mac(x, x, x)
    np.testing.assert_allclose(out, before * before + before, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(out, x)


def test_fortran_order_rejected():
    a = np.asfortranarray(np.ones((2, 3, 4)))
    b = np.ones((2, 3, 4))
    assert not a.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(a, b, b)


def test_transposed_view_rejected():
    a = np.ones((2, 3, 4)).transpose(2, 1, 0)
    b = np.ones((4, 3, 2))
    with pytest.raises(ValueError):
        mac(a, b, b)


def test_4d_rejected_even_if_squeezable():
    a = np.ones((1, 2, 3, 4))
    b = np.ones((2, 3, 4))
    with pytest.raises(ValueError):
        mac(a, b, b)


def test_scalar_float_rejected():
    b = np.ones((1, 1, 1))
    with pytest.raises(TypeError):
        mac(1.0, b, b)


def test_long_axis():
    rng = np.random.default_rng(7)
    a, b, c = [rng.uniform(-5, 5, size=(1, 1, 10_001)) for _ in range(3)]
    np.testing.assert_allclose(mac(a, b, c), a * b + c, rtol=1e-12, atol=1e-12)
