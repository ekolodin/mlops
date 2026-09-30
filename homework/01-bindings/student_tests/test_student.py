"""Extra checks for the installed tensor_ops.mac contract."""
import importlib

import numpy as np
import pytest


@pytest.fixture
def mac():
    return importlib.import_module("tensor_ops").mac


def test_numerical_unusual_shape(mac):
    shape = (1, 5, 1)
    a = np.array([[[2.0], [-1.5], [0.0], [4.0], [-0.25]]], dtype=np.float64)
    b = np.array([[[0.5], [2.0], [9.0], [-3.0], [8.0]]], dtype=np.float64)
    c = np.array([[[1.0], [-4.0], [3.0], [0.5], [0.125]]], dtype=np.float64)
    assert a.shape == shape
    expected = np.array([[[2.0], [-7.0], [3.0], [-11.5], [-1.875]]])
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_fortran_layout_is_rejected(mac):
    a = np.zeros((2, 3, 4), dtype=np.float64, order="F")
    b = np.ones((2, 3, 4), dtype=np.float64)
    c = np.full((2, 3, 4), 0.25, dtype=np.float64)
    assert not a.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(a, b, c)


def test_same_object_is_not_aliased_to_output(mac):
    value = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
    before = value.copy()
    out = mac(value, value, value)
    expected = before * before + before
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(value, before)
    assert not np.shares_memory(value, out)
    out[0, 0, 0] = 999.0
    np.testing.assert_array_equal(value, before)
