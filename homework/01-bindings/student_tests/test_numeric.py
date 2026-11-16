import pytest
import importlib
import importlib.util
import numpy as np

@pytest.fixture
def mac():
    assert importlib.util.find_spec("tensor_ops"), "Build and install your tensor_ops wheel first"
    package = importlib.import_module("tensor_ops")
    assert callable(getattr(package, "mac", None)), "Export tensor_ops.mac(a, b, c)"
    return package.mac

def test_same_args(mac):
    a = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[12.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 14.0, 3.0, 4.0, 5.0]]])
    a_copy = a.copy()
    out = mac(a, a, a)
    np.testing.assert_allclose(out, a ** 2 + a, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a, a_copy)
    assert not np.shares_memory(out, a)

    out.fill(-1.0)
    np.testing.assert_array_equal(a, a_copy)

def test_good_numbers(mac):
    a = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 2.0, 3.0, 4.0, 5.0]]])
    b = np.array([[[1.0, -1.0, 1.0, -1.0, 1.0]],
                  [[-1.0, 1.0, -1.0, 1.0, -1.0]],
                  [[1.0, -1.0, 1.0, -1.0, 1.0]]])
    c = np.array([[[10.0, 20.0, 30.0, 40.0, 50.0]],
                  [[10.0, 20.0, 30.0, 40.0, 50.0]],
                  [[10.0, 20.0, 30.0, 40.0, 50.0]]])

    res = np.array([[[11., 18., 33., 36., 55.]],
                    [[ 9., 22., 27., 44., 45.]],
                    [[11., 18., 33., 36., 55.]]])
    np.testing.assert_array_equal(mac(a, b, c), res)

    
