"""Additional checks against the installed tensor_ops package."""

import gc

import numpy as np
import pytest

from tensor_ops import mac


def test_fractional_values_with_singleton_axis():
    a = np.array([[[0.5, -1.5, 2.0]], [[-0.25, 0.0, 4.0]]])
    b = np.array([[[4.0, 2.0, -0.5]], [[8.0, -3.0, 0.25]]])
    c = np.array([[[1.0, 0.5, 3.0]], [[-1.0, 7.0, -2.0]]])
    expected = np.array([[[3.0, -2.5, 2.0]], [[-3.0, 7.0, -1.0]]])
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("layout", ["fortran", "transpose", "reverse"])
def test_unsupported_layout_in_each_argument(position, layout):
    args = [np.ones((2, 3, 4)) for _ in range(3)]
    if layout == "fortran":
        bad = np.asfortranarray(np.ones((2, 3, 4)))
    elif layout == "transpose":
        bad = np.ones((3, 2, 4)).transpose(1, 0, 2)
    else:
        bad = np.ones((2, 3, 4))[:, :, ::-1]
    assert bad.shape == args[0].shape
    assert not bad.flags.c_contiguous
    args[position] = bad
    with pytest.raises(ValueError):
        mac(*args)


def test_same_readonly_array_for_all_inputs():
    x = np.array([[[-2.0, 0.0, 0.5, 3.0]]])
    before = x.copy()
    x.flags.writeable = False
    out = mac(x, x, x)
    np.testing.assert_allclose(out, [[[2.0, 0.0, 0.75, 12.0]]], rtol=1e-12, atol=1e-12)
    assert out.flags.owndata and out.flags.c_contiguous
    assert not np.shares_memory(out, x)
    out.fill(-100.0)
    np.testing.assert_array_equal(x, before)


def test_result_survives_input_deletion():
    a = np.full((1, 2, 1), 2.0)
    b = np.full((1, 2, 1), -3.0)
    c = np.full((1, 2, 1), 0.5)
    out = mac(a, b, c)
    del a, b, c
    gc.collect()
    np.testing.assert_array_equal(out, [[[-5.5], [-5.5]]])


def test_equal_but_distinct_dtype_is_accepted():
    dtype = np.dtype("float64", copy=True)
    assert dtype == np.dtype("float64")
    assert dtype is not np.dtype("float64")
    a = np.full((1, 1, 2), 2.0, dtype=dtype)
    np.testing.assert_array_equal(mac(a, a, a), [[[6.0, 6.0]]])


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("invalid", ["dtype", "rank", "shape"])
def test_empty_inputs_are_still_validated(position, invalid):
    args = [np.empty((0, 2, 3)) for _ in range(3)]
    if invalid == "dtype":
        args[position] = np.empty((0, 2, 3), dtype=np.float32)
        error = TypeError
    elif invalid == "rank":
        args[position] = np.empty((0, 6))
        error = ValueError
    else:
        args[position] = np.empty((0, 3, 2))
        error = ValueError
    with pytest.raises(error):
        mac(*args)


def test_all_axes_empty():
    x = np.empty((0, 0, 0))
    out = mac(x, x, x)
    assert out.shape == (0, 0, 0)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous and out.flags.owndata
    assert out is not x
