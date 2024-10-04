import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sketchyopts import util


def test_default_types():
    """
    Test default_floating_dtype and default_integer_dtype.
    """
    # verify single precision types
    with jax.experimental.disable_x64():
        assert util.default_floating_dtype() == jnp.float32
        assert util.default_integer_dtype() == jnp.int32

    # verify double precision types
    with jax.experimental.enable_x64():
        assert util.default_floating_dtype() == jnp.float64
        assert util.default_integer_dtype() == jnp.int64


def test_asarray_no_copy():
    """
    Test inexact_asarray and integer_asarray and expect the converted array is the
    original array (not a copy).
    """
    # direct transformation
    x_float = jnp.array([1.0])
    x_int = jnp.array([1])
    assert util.inexact_asarray(x_float) is x_float
    assert util.integer_asarray(x_int) is x_int

    # vectorized transformation
    y_float = jnp.array([1.0, 2.0])
    y_int = jnp.array([1, 2])
    assert jax.vmap(util.inexact_asarray)(y_float) is y_float
    assert jax.vmap(util.integer_asarray)(y_int) is y_int


def test_asarray_result():
    """
    Test inexact_asarray and integer_asarray and expect the converted array has the
    correct data type.
    """
    # define examples
    objs = [1, 1.0, True, jnp.array([1]), jnp.array(1.0), np.array(1.0), np.array(1)]

    # transform examples
    for o in objs:
        assert util.inexact_asarray(o) == jnp.array(1.0)
        assert util.integer_asarray(o) == jnp.array(1.0)
