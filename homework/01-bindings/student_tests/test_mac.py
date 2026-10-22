import pytest
import numpy as np
from tensor_ops import mac

def test_numbers():
    a = np.array([[[2.0, -1.0, 0.0]]])
    b = np.array([[[3.0, 5.0, -2.0]]])
    c = np.array([[[1.0, 2.0, 7.0]]])
    expected = np.array([[[7.0, -3.0, 7.0]]])
    result = mac(a, b, c)

    np.testing.assert_array_equal(result, expected)

def test_invalid_dtype():
    a = np.ones((2, 3, 4), dtype=np.float32)
    b = np.ones((2, 3, 4), dtype=np.float64)
    c = np.ones((2, 3, 4), dtype=np.float64)

    with pytest.raises(TypeError, match="float64"):
        mac(a, b, c)

def test_memory_independence():
    a = np.array([[[1.0, 2.0, 3.0]]])
    b = np.array([[[4.0, 5.0, 6.0]]])
    c = np.array([[[7.0, 8.0, 9.0]]])
    a_before = a.copy()
    b_before = b.copy()
    c_before = c.copy()
    result = mac(a, b, c)

    assert not np.shares_memory(result, a)
    assert not np.shares_memory(result, b)
    assert not np.shares_memory(result, c)

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
    np.testing.assert_array_equal(c, c_before)

    result[0, 0, 0] = 999.0

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
    np.testing.assert_array_equal(c, c_before)