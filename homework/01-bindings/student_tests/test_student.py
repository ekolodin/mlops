import numpy as np
import pytest


def test_numeric_example_matches_numpy():
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = np.full((2, 3, 4), 2.0, dtype=np.float64)
    c = np.full((2, 3, 4), -1.0, dtype=np.float64)
    import importlib.util

    assert importlib.util.find_spec("tensor_ops"), "Build and install the wheel first"
    from tensor_ops import mac

    expected = a * b + c
    np.testing.assert_allclose(mac(a, b, c), expected, rtol=1e-12, atol=1e-12)


def test_rejects_wrong_shape_and_raises_value_error():
    from tensor_ops import mac

    a = np.ones((2, 3, 4), dtype=np.float64)
    b = np.ones((1, 3, 4), dtype=np.float64)
    c = np.ones((2, 3, 4), dtype=np.float64)

    with pytest.raises(ValueError):
        mac(a, b, c)


def test_input_views_do_not_alias_output():
    from tensor_ops import mac

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = np.full((2, 3, 4), 0.5, dtype=np.float64)
    c = np.full((2, 3, 4), -2.0, dtype=np.float64)
    out = mac(a, b, c)

    assert out.shape == a.shape
    assert out.dtype == np.dtype("float64")
    assert not np.shares_memory(out, a)
    assert not np.shares_memory(out, b)
    assert not np.shares_memory(out, c)
