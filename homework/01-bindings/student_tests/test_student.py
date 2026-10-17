import numpy as np
import pytest

from tensor_ops import mac


def test_known_values_on_nontrivial_shape():
    a = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    b = np.array([[[2.0, -1.0], [0.5, 0.0], [-2.0, 10.0]]])
    c = np.array([[[0.0, 1.0], [-1.5, 7.0], [3.0, -60.0]]])
    expected = np.array([[[2.0, -1.0], [0.0, 7.0], [-7.0, 0.0]]])
    out = mac(a, b, c)
    assert out.shape == (1, 3, 2)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_same_object_for_all_inputs():
    x = np.full((2, 2, 2), 3.0)
    snapshot = x.copy()
    out = mac(x, x, x)
    np.testing.assert_array_equal(out, np.full((2, 2, 2), 12.0))
    np.testing.assert_array_equal(x, snapshot)
    assert not np.shares_memory(out, x)


def test_fortran_layout_is_rejected():
    a = np.asfortranarray(np.ones((2, 3, 4)))
    assert not a.flags.c_contiguous
    b = np.ones((2, 3, 4))
    with pytest.raises(ValueError):
        mac(a, b, b)


def test_transposed_view_is_rejected():
    a = np.ones((4, 3, 2)).transpose(2, 1, 0)
    b = np.ones((2, 3, 4))
    with pytest.raises(ValueError):
        mac(a, b, b)


def test_scalar_and_zero_dim_array_are_rejected():
    b = np.ones((1, 1, 1))
    with pytest.raises(TypeError):
        mac(1.0, b, b)
    with pytest.raises(ValueError):
        mac(np.array(1.0), b, b)


def test_empty_inputs_with_wrong_dtype_are_still_rejected():
    a = np.empty((0, 3, 4), dtype=np.float32)
    b = np.empty((0, 3, 4))
    with pytest.raises(TypeError):
        mac(a, b, b)


def test_empty_inputs_with_mismatched_shapes_are_rejected():
    a = np.empty((0, 3, 4))
    b = np.empty((0, 4, 3))
    with pytest.raises(ValueError):
        mac(a, b, b)
