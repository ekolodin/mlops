import numpy as np
import pytest

from tensor_ops import mac


def test_values():
    a = np.array([[[2.0, -3.0]], [[4.0, 0.5]]])
    b = np.array([[[-1.0, 2.0]], [[3.0, -4.0]]])
    c = np.array([[[5.0, 1.0]], [[-2.0, 7.0]]])

    np.testing.assert_array_equal(mac(a, b, c), [[[3.0, -5.0]], [[10.0, 5.0]]])


def test_fortran_input_is_rejected():
    a = np.zeros((2, 3, 4), dtype=np.float64)
    b = np.asfortranarray(np.ones_like(a))
    assert not b.flags.c_contiguous

    with pytest.raises(ValueError):
        mac(a, b, a)


def test_output_does_not_share_memory_with_readonly_input():
    source = np.array([[[1.0, 2.0, 3.0]]])
    source.flags.writeable = False
    offset = np.ones_like(source)

    result = mac(source, source, offset)
    np.testing.assert_array_equal(result, [[[2.0, 5.0, 10.0]]])
    assert not np.shares_memory(result, source)
    assert not np.shares_memory(result, offset)
    result.fill(0.0)
    np.testing.assert_array_equal(source, [[[1.0, 2.0, 3.0]]])
    np.testing.assert_array_equal(offset, [[[1.0, 1.0, 1.0]]])
