"""Additional checks for the installed tensor_ops wheel."""

import numpy as np
import pytest

from tensor_ops import mac


def test_nonsquare_negative_values_match_independent_expression():
    a = np.array([[[-3.5, 0.0], [2.0, 9.0]]], dtype=np.float64)
    b = np.array([[[4.0, -8.0], [0.5, -1.0]]], dtype=np.float64)
    c = np.array([[[1.0, 2.0], [-6.0, 0.25]]], dtype=np.float64)

    np.testing.assert_allclose(mac(a, b, c), a * b + c, rtol=0, atol=0)


def test_fortran_order_input_is_rejected():
    a = np.asfortranarray(np.ones((2, 3, 4), dtype=np.float64))
    b = np.ones((2, 3, 4), dtype=np.float64)
    c = np.ones((2, 3, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="C-contiguous"):
        mac(a, b, c)


def test_same_input_object_is_read_only_and_output_is_independent():
    source = np.arange(8, dtype=np.float64).reshape(1, 2, 4)
    source.flags.writeable = False

    result = mac(source, source, source)

    np.testing.assert_array_equal(result, source * source + source)
    assert not np.shares_memory(result, source)
