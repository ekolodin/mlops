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

def test_transpose(mac):
    a = np.array([[[1.,  6.],
                   [2.,  7.],
                   [3.,  8.],
                   [4.,  9.],
                   [5., 10.]]])
    b = np.array([[[1.], [2.], [3.], [4.], [5.]],
                  [[6.], [7.], [8.], [9.], [10.]]])

    with pytest.raises(ValueError):
        mac(a, b.T, b.T)
    