import numpy as np
import pytest

from tensor_ops import mac


def test_custom_numerical_case():
    a = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float64,
    )

    b = np.full(a.shape, 2.0, dtype=np.float64)
    c = np.full(a.shape, 1.0, dtype=np.float64)

    expected = a * 2.0 + 1.0

    result = mac(a, b, c)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_fortran_order_is_rejected():
    shape = (2, 3, 4)

    a = np.asfortranarray(
        np.arange(24, dtype=np.float64).reshape(shape)
    )
    b = np.ones(shape, dtype=np.float64)
    c = np.ones(shape, dtype=np.float64)

    assert not a.flags.c_contiguous

    with pytest.raises(ValueError):
        mac(a, b, c)


def test_empty_array_custom_case():
    shape = (3, 2, 0)

    a = np.empty(shape, dtype=np.float64)
    b = np.empty(shape, dtype=np.float64)
    c = np.empty(shape, dtype=np.float64)

    result = mac(a, b, c)

    assert result.shape == shape
    assert result.size == 0
    assert result.dtype == np.float64
    assert result.flags.c_contiguous