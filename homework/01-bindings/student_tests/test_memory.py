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

def test_independent_results(mac):
    a = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 2.0, 3.0, 4.0, 5.0]]])
    b = np.array([[[1.0, -1.0, 1.0, -1.0, 1.0]],
                  [[-1.0, 1.0, -1.0, 1.0, -1.0]],
                  [[1.0, -1.0, 1.0, -1.0, 1.0]]])
    c = np.array([[[10.0, 20.0, 30.0, 40.0, 50.0]],
                  [[10.0, 20.0, 30.0, 40.0, 50.0]],
                  [[10.0, 20.0, 30.0, 40.0, 50.0]]])

    out1 = mac(a, b, c)
    out2 = mac(a, b, c)
    np.testing.assert_array_equal(out1, out2)
    assert not np.shares_memory(out1, out2)

def test_named_args(mac):
    a = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 2.0, 3.0, 4.0, 5.0]],
                  [[1.0, 2.0, 3.0, 4.0, 5.0]]])
    b = np.array([[[1.0, -1.0, 1.0, -1.0, 1.0]],
                  [[-1.0, 1.0, -1.0, 1.0, -1.0]],
                  [[1.0, -1.0, 1.0, -1.0, 1.0]]])
    c = np.array([[[10.0, 20.0, 30.0, 40.0, 50.0]],
                  [[10.0, 20.0, 30.0, 40.0, 50.0]],
                  [[10.0, 20.0, 30.0, 40.0, 50.0]]])

    out_positional = mac(a.copy(), b.copy(), c.copy())
    out_keyword = mac(a=a.copy(), b=b.copy(), c=c.copy())
    np.testing.assert_array_equal(out_positional, out_keyword)