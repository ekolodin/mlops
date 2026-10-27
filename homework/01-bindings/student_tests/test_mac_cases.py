import numpy as np
import pytest

from tensor_ops import mac


def test_mac_with_same_input_object_and_expected_values():
    x = np.array(
        [
            [
                [-2.0, -1.5, 0.0],
                [1.25, 2.0, 3.5],
            ],
            [
                [4.0, -3.0, 0.5],
                [10.0, 0.25, -8.0],
            ],
        ],
        dtype=np.float64,
    )
    x_before = x.copy()

    result = mac(x, x, x)

    expected = x_before * x_before + x_before

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, x_before)

    assert result.dtype == np.float64
    assert result.shape == x.shape
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, x)


def test_mac_accepts_read_only_inputs_without_mutating_them():
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = np.full((2, 3, 4), -2.5, dtype=np.float64)
    c = np.linspace(-1.0, 1.0, num=24, dtype=np.float64).reshape(2, 3, 4)

    a_before = a.copy()
    b_before = b.copy()
    c_before = c.copy()

    a.setflags(write=False)
    b.setflags(write=False)
    c.setflags(write=False)

    result = mac(a, b, c)

    expected = a_before * b_before + c_before

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
    np.testing.assert_array_equal(c, c_before)

    assert result.flags.writeable
    assert not np.shares_memory(result, a)
    assert not np.shares_memory(result, b)
    assert not np.shares_memory(result, c)


def test_mac_empty_axis_returns_independent_empty_array():
    a = np.empty((2, 0, 4), dtype=np.float64)
    b = np.empty((2, 0, 4), dtype=np.float64)
    c = np.empty((2, 0, 4), dtype=np.float64)

    result = mac(a, b, c)

    assert result.shape == (2, 0, 4)
    assert result.size == 0
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, a)
    assert not np.shares_memory(result, b)
    assert not np.shares_memory(result, c)


@pytest.mark.parametrize(
    "bad_input",
    [
        np.asfortranarray(
            np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        ),
        np.arange(48, dtype=np.float64).reshape(2, 3, 8)[:, :, ::2],
        np.arange(24, dtype=np.float64)
        .reshape(4, 3, 2)
        .transpose(2, 1, 0),
    ],
    ids=[
        "fortran_layout",
        "strided_slice",
        "transposed_view",
    ],
)
def test_mac_rejects_non_c_contiguous_inputs(bad_input):
    good = np.ones((2, 3, 4), dtype=np.float64)

    assert bad_input.shape == good.shape
    assert not bad_input.flags.c_contiguous

    with pytest.raises(ValueError):
        mac(bad_input, good, good)

    with pytest.raises(ValueError):
        mac(good, bad_input, good)

    with pytest.raises(ValueError):
        mac(good, good, bad_input)