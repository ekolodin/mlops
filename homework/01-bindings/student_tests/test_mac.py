import numpy as np
import pytest

from tensor_ops import mac


def test_unusual_shape_and_equal_dtype_object():
    a = np.array([[[-4.0, 1.5, 7.0]]], dtype=np.dtype("float64", copy=True))
    b = np.array([[[2.0, -3.0, 0.25]]])
    c = np.array([[[9.0, 2.0, -1.0]]])
    np.testing.assert_array_equal(mac(a, b, c), [[[1.0, -2.5, 0.75]]])


def test_fortran_layout_is_rejected():
    a = np.asfortranarray(np.arange(-12.0, 12.0).reshape(2, 3, 4))
    b = np.full((2, 3, 4), 3.0)
    with pytest.raises(ValueError):
        mac(a, b, b)


def test_aliases_and_readonly_inputs_do_not_share_output():
    a = np.array([[[4.0, -1.0]]])
    a.flags.writeable = False
    out = mac(a, a, a)
    np.testing.assert_array_equal(out, [[[20.0, 0.0]]])
    assert not np.shares_memory(out, a)
    out.fill(99.0)
    np.testing.assert_array_equal(a, [[[4.0, -1.0]]])
