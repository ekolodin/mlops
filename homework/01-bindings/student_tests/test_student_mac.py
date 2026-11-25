"""Student tests for tensor_ops.mac. Run against an installed wheel, like tests/."""
import importlib
import importlib.util

import numpy as np
import pytest

SHAPE = (2, 3, 4)


@pytest.fixture
def mac():
    assert importlib.util.find_spec("tensor_ops"), "Build and install your tensor_ops wheel first"
    return importlib.import_module("tensor_ops").mac


# Numeric: expected values are computed by hand or with plain Python, not with mac.


def test_same_array_for_all_inputs(mac):
    x = np.array([[[1.0, 2.0, -3.0], [0.5, 0.0, -1.0]]])
    before = x.copy()
    out = mac(x, x, x)
    # x * x + x for each element
    np.testing.assert_allclose(out, [[[2.0, 6.0, 6.0], [0.75, 0.0, 0.0]]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(out, x)


def test_each_element_keeps_its_index(mac):
    a = np.empty(SHAPE)
    for i, j, k in np.ndindex(SHAPE):
        a[i, j, k] = 100 * i + 10 * j + k
    out = mac(a, np.full(SHAPE, 2.0), np.full(SHAPE, -1.0))
    for i, j, k in np.ndindex(SHAPE):
        assert out[i, j, k] == 2 * (100 * i + 10 * j + k) - 1


@pytest.mark.parametrize("shape", [(1, 1, 7), (7, 1, 1), (1, 5, 1), (3, 1, 2)])
def test_unusual_shapes(mac, shape):
    rng = np.random.default_rng(7)
    a, b, c = (rng.uniform(-10.0, 10.0, size=shape) for _ in range(3))
    out = mac(a, b, c)
    assert out.shape == shape
    np.testing.assert_allclose(out, a * b + c, rtol=1e-12, atol=1e-12)


def test_equal_but_distinct_dtype_object_is_accepted(mac):
    a = np.ones(SHAPE, dtype=np.dtype("float64", copy=True))
    assert a.dtype is not np.dtype(np.float64)
    np.testing.assert_allclose(mac(a, a, a), np.full(SHAPE, 2.0), rtol=1e-12, atol=1e-12)


# Negative: invalid input is rejected, not converted or copied.


def fortran_order():
    return np.asfortranarray(np.ones(SHAPE))


def permuted_axes():
    return np.ones((3, 2, 4)).transpose(1, 0, 2)


def negative_stride():
    return np.ones(SHAPE)[::-1]


@pytest.mark.parametrize("make", [fortran_order, permuted_axes, negative_stride])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_non_c_contiguous_layouts_are_rejected(mac, make, position):
    args = [np.ones(SHAPE) for _ in range(3)]
    args[position] = make()
    assert args[position].shape == SHAPE and not args[position].flags.c_contiguous
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("value", [None, 1.0, np.float64(1.0), (1.0, 2.0)],
                         ids=["none", "float", "numpy-scalar", "tuple"])
def test_non_array_arguments_are_rejected(mac, value):
    with pytest.raises(TypeError):
        mac(np.ones(SHAPE), value, np.ones(SHAPE))


# Memory and boundary cases.


def test_inputs_are_slices_of_one_buffer(mac):
    buffer = np.arange(3 * 24, dtype=np.float64).reshape(3, *SHAPE)
    before = buffer.copy()
    a, b, c = buffer
    out = mac(a, b, c)
    np.testing.assert_allclose(out, before[0] * before[1] + before[2], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(buffer, before)
    assert not np.shares_memory(out, buffer)


def test_output_owns_its_memory(mac):
    a, b, c = np.full(SHAPE, 2.0), np.full(SHAPE, 3.0), np.full(SHAPE, 1.0)
    out = mac(a, b, c)
    assert out.flags.owndata and out.flags.writeable and out.flags.c_contiguous
    a[...] = b[...] = c[...] = 0.0
    np.testing.assert_array_equal(out, np.full(SHAPE, 7.0))
    assert not np.shares_memory(out, mac(a, b, c))


@pytest.mark.parametrize(
    "bad, error",
    [
        (np.zeros((0, 3, 4), dtype=np.float32), TypeError),
        (np.zeros((0, 4, 3)), ValueError),
        (np.zeros((0, 12)), ValueError),
    ],
    ids=["float32", "other-shape", "2d"],
)
def test_empty_inputs_are_still_validated(mac, bad, error):
    args = [np.zeros((0, 3, 4)) for _ in range(3)]
    args[1] = bad
    with pytest.raises(error):
        mac(*args)
