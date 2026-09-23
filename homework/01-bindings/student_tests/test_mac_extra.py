import numpy as np
import pytest

from tensor_ops import mac


def test_mac_with_independently_calculated_values():
    a = np.array([[[1.0, -2.0], [0.5, 4.0]]])
    b = np.array([[[3.0, 2.0], [-2.0, 0.25]]])
    c = np.array([[[0.5, 1.0], [4.0, -2.0]]])

    expected = np.array([[[3.5, -3.0], [3.0, -1.0]]])

    np.testing.assert_allclose(
        mac(a, b, c),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_fortran_contiguous_input_is_rejected():
    a = np.asfortranarray(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    )
    b = np.ones((2, 3, 4), dtype=np.float64)
    c = np.zeros((2, 3, 4), dtype=np.float64)

    assert a.flags.f_contiguous
    assert not a.flags.c_contiguous

    with pytest.raises(ValueError):
        mac(a, b, c)


def test_same_input_object_is_not_modified_or_shared():
    value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    before = value.copy()

    result = mac(value, value, value)

    np.testing.assert_array_equal(value, before)
    np.testing.assert_allclose(result, before * before + before)
    assert not np.shares_memory(result, value)