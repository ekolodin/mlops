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


def test_same_object_for_all_inputs(mac):
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    expected = a * a + a
    result = mac(a, a, a)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_fortran_order_is_rejected(mac):
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    f_order = np.asfortranarray(a)
    assert not f_order.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(f_order, a, a)

def test_mismatch_in_last_axis_is_rejected(mac):
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    wrong = np.zeros((2, 3, 5), dtype=np.float64)
    assert wrong.shape != a.shape
    with pytest.raises(ValueError):
        mac(a, wrong, a)