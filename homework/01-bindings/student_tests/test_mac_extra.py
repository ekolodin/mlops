import numpy as np
import pytest

from tensor_ops import mac


def test_product_cancellation():
    a = np.full((2, 3, 4), 200.1)
    b = np.full((2, 3, 4), 200.1)
    c = -(a * b)
    expected = a * b + c
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_fractional_values():
    a = np.array([[[-2.5], [0.125], [4.0], [-0.5]]])
    b = np.array([[[0.4], [-8.0], [-0.25], [-6.0]]])
    c = np.array([[[3.0], [0.5], [1.5], [-2.0]]])
    expected = np.array([[[2.0], [-0.5], [0.5], [1.0]]])
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_same_readonly_input():
    x = np.array([[[-2.0, 0.0, 0.5, 3.0]]])
    before = x.copy()
    x.flags.writeable = False
    out = mac(x, x, x)
    np.testing.assert_allclose(out, [[[2.0, 0.0, 0.75, 12.0]]], rtol=1e-12, atol=1e-12)
    assert out.flags.owndata and out.flags.writeable
    assert out.flags.c_contiguous and out.flags.aligned
    assert not np.shares_memory(out, x)
    out.fill(99.0)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("position", range(3))
def test_equal_but_distinct_dtype(position):
    values = (2.0, 3.0, -1.0)
    args = [np.full((1, 2, 3), value) for value in values]
    dtype = np.dtype("float64", copy=True)
    assert dtype == np.dtype("float64") and dtype is not np.dtype("float64")
    args[position] = np.full((1, 2, 3), values[position], dtype=dtype)
    assert args[position].dtype is dtype
    np.testing.assert_allclose(mac(*args), np.full((1, 2, 3), 5.0), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("layout", ["fortran", "transpose", "negative_stride"])
def test_invalid_layout(position, layout):
    args = [np.ones((2, 3, 4)) for _ in range(3)]
    if layout == "fortran":
        args[position] = np.asfortranarray(args[position])
    elif layout == "transpose":
        args[position] = np.ones((4, 3, 2)).transpose(2, 1, 0)
    else:
        args[position] = args[position][:, :, ::-1]
    assert args[position].shape == (2, 3, 4)
    assert not args[position].flags.c_contiguous
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("violation", ["dtype", "byte_order", "rank", "shape"])
def test_invalid_empty_arrays(position, violation):
    args = [np.empty((0, 2, 3)) for _ in range(3)]
    error = ValueError
    if violation == "dtype":
        args[position] = np.empty((0, 2, 3), dtype=np.float32)
        error = TypeError
    elif violation == "byte_order":
        args[position] = np.empty((0, 2, 3), dtype=np.dtype("float64").newbyteorder("S"))
        error = TypeError
    elif violation == "rank":
        args[position] = np.empty((0, 6))
    else:
        args[position] = np.empty((0, 3, 2))
    with pytest.raises(error):
        mac(*args)


def test_output_outlives_inputs():
    a = np.full((2, 1, 3), 1.5)
    b = np.full((2, 1, 3), 2.0)
    c = np.full((2, 1, 3), -4.0)
    out = mac(a, b, c)
    del a, b, c
    assert out.flags.owndata
    np.testing.assert_array_equal(out, np.full((2, 1, 3), -1.0))
