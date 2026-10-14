import numpy as np
import pytest

from tensor_ops import mac


def test_numeric_hand_calculated():
    a = np.array([[[2.0, 3.0]]])
    b = np.array([[[4.0, -1.0]]])
    c = np.array([[[1.0, 1.0]]])
    np.testing.assert_allclose(mac(a, b, c), [[[9.0, -2.0]]], rtol=1e-12, atol=1e-12)


def test_fortran_layout_is_rejected():
    a = np.asfortranarray(np.ones((2, 3, 4)))
    b = np.ones((2, 3, 4))
    c = np.ones((2, 3, 4))
    assert not a.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(a, b, c)


def test_same_object_for_all_inputs():
    x = np.full((1, 2, 2), 3.0)
    before = x.copy()
    out = mac(x, x, x)
    np.testing.assert_allclose(out, np.full((1, 2, 2), 12.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(out, x)