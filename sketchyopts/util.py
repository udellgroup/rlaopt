import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sketchyopts import tree_util

KeyArray = Array
KeyArrayLike = ArrayLike

ravel_tree = tree_util.ravel_tree
tree_flatten = tree_util.tree_flatten
tree_unflatten = tree_util.tree_unflatten
tree_leaves = tree_util.tree_leaves
tree_add = tree_util.tree_add
tree_sub = tree_util.tree_sub
tree_scalar_mul = tree_util.tree_scalar_mul
tree_add_scalar_mul = tree_util.tree_add_scalar_mul
tree_l2_norm = tree_util.tree_l2_norm


def default_floating_dtype():
    r"""Get default floating dtype.

    Returns:
      If double-precision mode is enabled in JAX, the function returns ``float64``, otherwise ``float32``.
    """
    if jax.config.jax_enable_x64:  # type: ignore
        return jnp.float64
    else:
        return jnp.float32


def default_integer_dtype():
    r"""Get default integer dtype.

    Returns:
      If double-precision mode is enabled in JAX, the function returns ``int64``, otherwise ``int32``.
    """
    if jax.config.jax_enable_x64:  # type: ignore
        return jnp.int64
    else:
        return jnp.int32


def inexact_asarray(x):
    r"""Convert the input array to an array of explicitly specified inexact dtype.

    The function converts the input to an array of default floating dtype if the current dtype of the input is not an inexact type.
    Otherwise the function converts the array to a strongly-typed one.

    Args:
      x: Input array.

    Returns:
      Converted strongly-typed array of inexact type.

    See Also:
        :class:`jax.numpy.inexact`
    """
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = default_floating_dtype()
    return jnp.asarray(x, dtype=dtype)


def integer_asarray(x):
    r"""Convert the input scalar to an scalar of explicitly specified integer dtype.

    The function converts the input to an scalar of default integer dtype if the current dtype of the input is not an integer type.
    Otherwise the function converts the scalar to a strongly-typed one.

    Args:
      x: Input scalar.

    Returns:
      Converted strongly-typed scalar of integer type.

    See Also:
        :class:`jax.numpy.integer`
    """
    dtype = jnp.result_type(x)
    if not jnp.issubdtype(dtype, jnp.integer):
        dtype = default_integer_dtype()
    return jnp.asarray(x, dtype=dtype)
