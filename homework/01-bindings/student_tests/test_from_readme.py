import importlib
import importlib.util

import numpy as np
import pytest


@pytest.fixture
def tensor_ops():
    assert importlib.util.find_spec("tensor_ops"), (
        "Build and install your tensor_ops wheel first"
    )
    return importlib.import_module("tensor_ops")


def test_values_on_asymmetric_shape(tensor_ops):
    a = np.array([[[2.0, -1.0], [0.5, 3.0]]])
    b = np.array([[[4.0, 2.0], [-8.0, 0.25]]])
    c = np.array([[[1.0, 5.0], [2.0, -4.0]]])

    out = tensor_ops.mac(a, b, c)

    expected = np.array([[[9.0, 3.0], [-2.0, -3.25]]])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_fortran_layout_is_rejected(tensor_ops):
    a = np.asfortranarray(np.ones((2, 3, 4), dtype=np.float64))
    b = np.ones((2, 3, 4), dtype=np.float64)
    c = np.ones((2, 3, 4), dtype=np.float64)

    assert a.flags.f_contiguous
    assert not a.flags.c_contiguous
    with pytest.raises(ValueError):
        tensor_ops.mac(a, b, c)


def test_aliasing_inputs_do_not_change_input_or_share_output(tensor_ops):
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    before = x.copy()

    out = tensor_ops.mac(x, x, x)

    np.testing.assert_array_equal(x, before)
    np.testing.assert_allclose(out, before * before + before)
    assert not np.shares_memory(out, x)
