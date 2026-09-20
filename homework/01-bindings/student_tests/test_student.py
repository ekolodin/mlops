"""Student tests"""
import importlib
import importlib.util

import numpy as np
import pytest


@pytest.fixture
def mac():
    assert importlib.util.find_spec("tensor_ops"), "Build and install your tensor_ops wheel first"
    package = importlib.import_module("tensor_ops")
    assert callable(getattr(package, "mac", None)), "Export tensor_ops.mac(a, b, c)"
    return package.mac


def test_large_shape_matches_numpy(mac):
    rng = np.random.default_rng(7)
    shape = (50, 60, 70)
    a, b, c = [rng.normal(size=shape) for _ in range(3)]
    expected = a * b + c
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_dtype_error_takes_priority_over_shape_error(mac):
    good = np.zeros((2, 3, 4), dtype=np.float64)
    bad_dtype = good.astype(np.float32)
    bad_shape = np.zeros((2, 3, 5), dtype=np.float64)
    with pytest.raises(TypeError):
        mac(bad_dtype, bad_shape, good)


def test_mismatch_in_last_axis_is_rejected(mac):
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    wrong = np.zeros((2, 3, 5), dtype=np.float64)
    assert wrong.shape != a.shape
    with pytest.raises(ValueError):
        mac(a, wrong, a)