import numpy as np
import pytest

from tensor_ops import mac


def test_hand_calculated_values_for_several_elements():
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.5], [10.0, -3.0]]])
    b = np.array([[[2.0, 2.0], [0.0, -1.0]], [[4.0, 4.0], [0.1, 3.0]]])
    c = np.array([[[0.0, 1.0], [5.0, 0.0]], [[1.0, -2.0], [-1.0, 9.0]]])
    expected = np.array([[[2.0, 5.0], [5.0, -4.0]], [[-3.0, 0.0], [0.0, 0.0]]])
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_fortran_layout_is_rejected():
    a = np.ones((2, 3, 4))
    f = np.asfortranarray(a)
    assert not f.flags.c_contiguous
    with pytest.raises(ValueError):
        mac(a, f, a)


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3, 4)])
def test_same_wrong_rank_for_all_inputs_is_rejected(shape):
    x = np.ones(shape)
    with pytest.raises(ValueError):
        mac(x, x, x)


def test_same_object_for_all_inputs():
    x = np.array([[[1.0, 2.0, -3.0]]])
    snapshot = x.copy()
    out = mac(x, x, x)
    np.testing.assert_allclose(out, [[[2.0, 6.0, 6.0]]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, snapshot)
    assert not np.shares_memory(out, x)
