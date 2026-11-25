import numpy as np
import tensor_ops as tops
import pytest

def test_mac_hand_calculated():
    a = np.array([[[2.0, -1.0]]], dtype=np.float64)
    b = np.array([[[3.0, 4.0]]], dtype=np.float64)
    c = np.array([[[1.0, 5.0]]], dtype=np.float64)
    result = np.array([[[7.0, 1.0]]], dtype=np.float64)

    np.testing.assert_array_equal(tops.mac(a, b, c), result)


def test_mac_rejects_non_array():
    a = "a"
    b = np.array([[[3.0, 4.0]]], dtype=np.float64)
    c = np.array([[[1.0, 5.0]]], dtype=np.float64)

    with pytest.raises(TypeError):
        tops.mac(a, b, c)


def test_mac_memory():
    x = np.array([[[2.0, -1.0]]], dtype=np.float64)
    x_copy = x.copy()
    expected = np.array([[[6.0, 0.0]]], dtype=np.float64)

    result = tops.mac(x, x, x)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(x, x_copy)
    assert not np.shares_memory(result, x)