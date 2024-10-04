# Pytree utilities in this module are adapted from JAXopt with modifications.
# - Documentation of the JAXopt tree utilities:
#   https://jaxopt.github.io/stable/api.html#tree-utilities
# - Original implementation:
#   https://github.com/google/jaxopt/blob/main/jaxopt/_src/tree_util.py
#
# Copyright license information:
#
# Copyright 2021 Google LLC
# Modifications copyright 2024 the SketchyOpts authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import itertools
import operator

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src import flatten_util

ravel_tree = flatten_util.ravel_pytree
ravel_tree.__annotations__ = {}
ravel_tree.__doc__ = r"""
Ravel a tree to a 1-dimensional array.

This function flattens a tree to a 1-dimensional array (*i.e.* the flattened and 
concatenated leaf values). An alias of :func:`jax.flatten_util.ravel_pytree`.

Args: 
  pytree: A pytree of arrays and scalars to ravel. 

Returns:
  A two-element tuple containing

  - **raveled_tree** – 1-dimensional array representing the flattened and concatenated 
    leaf values. 
  - **unravel_fun** – Callable for unflattening a 1-dimensional array back to a tree of 
    the same structure as ``pytree``.

See Also:
  :func:`jax.flatten_util.ravel_pytree`
  :func:`tree_flatten`
"""

tree_map = jtu.tree_map
tree_map.__annotations__ = {}
tree_map.__doc__ = r"""
Apply a function at the leaves of pytrees.  

This function maps a multi-input function over pytree arguments to produce a new pytree. 
An alias of :func:`jax.tree.map`.

Args: 
  f: A function be applied at the leaves of pytrees. 
  tree: A pytree to be mapped over. Each leaf of the pytree serves as the first 
    positional argument to the function ``f``. 
  rest: A tuple of pytrees. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf and thus does not flattening (default ``None``).

Returns:
  A new pytree with the same structure as ``tree`` but with the value at each leaf given 
  by ``f(tree_leaf, *rest_leaves)``. 

See Also:
  :func:`jax.tree.map`
"""

tree_reduce = jtu.tree_reduce
tree_reduce.__annotations__ = {}
tree_reduce.__doc__ = r"""Call ``reduce`` over the leaves of a tree."""

tree_flatten = jtu.tree_flatten
tree_flatten.__annotations__ = {}
tree_flatten.__doc__ = r"""
Flatten a tree to a list of leaves.

This function flattens a tree to the corresponding list of leaf values. An alias of 
:func:`jax.tree.flatten`.

Args: 
  tree: Pytree to flatten. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf and thus does not flattening (default ``None``). 

Returns:
  A two-element tuple containing

  - **leaves** – List of leaf values.
  - **treedef** – Treedef representing the structure of the flattened tree.
"""

tree_unflatten = jtu.tree_unflatten
tree_unflatten.__annotations__ = {}
tree_unflatten.__doc__ = r"""
Unflatten a tree from leaves. 

This function reconstructs a tree from the treedef and the leaves. An alias of 
:func:`jax.tree.unflatten`.

Args: 
  treedef: Treedef to use for tree reconstruction. 
  leaves: Matching iterable of leaves the tree reconstructs from. 

Returns:
  Reconstructed tree. 
"""

tree_leaves = jtu.tree_leaves
tree_leaves.__annotations__ = {}
tree_leaves.__doc__ = r"""
Get the leaves of a tree.

Similar to :func:`tree_flatten`, the function flattens a tree. The difference is that 
only a list of leaf values gets returned. An alias of :func:`jax.tree.leaves`.

Args: 
  tree: Pytree to flatten. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf and thus does not flattening (default ``None``). 

Returns:
  A list of tree leaves.
"""

tree_structure = jtu.tree_structure
tree_structure.__annotations__ = {}
tree_structure.__doc__ = r"""Get the treedef for a pytree."""


def broadcast_pytrees(*trees):
    r"""Broadcast leaf pytrees to match treedef shared by the other arguments.

    Args:
      *trees: A ``Sequence`` of pytrees such that all elements that are *not* leaf
        pytrees (*i.e.* single arrays) have the same treedef.

    Returns:
      The input ``Sequence`` of pytrees ``*trees`` with leaf pytrees (*i.e.* single
      arrays) replaced by pytrees matching the treedef of non-shallow elements via
      broadcasting.

    Raises:
      ValueError: If two or more pytrees in ``*trees`` that are not leaf pytrees differ
        in their structure (treedef).
    """
    leaves, treedef, is_leaf = [], None, []
    for tree in trees:
        leaves_i, treedef_i = jtu.tree_flatten(tree)
        is_leaf_i = jtu.treedef_is_leaf(treedef_i)
        if not is_leaf_i:
            treedef = treedef or treedef_i
            if treedef_i != treedef:
                raise ValueError(
                    "Pytrees are not broadcastable.: " f"{treedef} != {treedef_i}"
                )
        leaves.append(leaves_i)
        is_leaf.append(is_leaf_i)
    if treedef is not None:
        max_num_leaves = max(len(leaves_i) for leaves_i in leaves)
        broadcast_leaf = lambda leaf: itertools.repeat(leaf[0], max_num_leaves)
        leaves = [
            broadcast_leaf(leaves_i) if is_leaf_i else leaves_i
            for (leaves_i, is_leaf_i) in zip(leaves, is_leaf)
        ]
        return tuple(treedef.unflatten(leaves_i) for leaves_i in leaves)
    # All Pytrees are leaves.
    return trees


tree_add = functools.partial(tree_map, operator.add)
tree_add.__doc__ = r"""
Compute tree addition.

This function computes :math:`tree + rest`.

Args: 
  tree: Pytree that precedes the addition operator. 
  rest: Pytree that succeeds the addition operator. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf (default ``None``). 

Returns:
  Resulting pytree. 
"""

tree_sub = functools.partial(tree_map, operator.sub)
tree_sub.__doc__ = r"""
Compute tree subtraction.

This function computes :math:`tree - rest`.

Args: 
  tree: Pytree that precedes the subtraction operator. 
  rest: Pytree that succeeds the subtraction operator. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf (default ``None``). 

Returns:
  Resulting pytree. 
"""

tree_mul = functools.partial(tree_map, operator.mul)
tree_mul.__doc__ = r"""
Compute tree multiplication.

This function computes :math:`tree \odot rest`.

Args: 
  tree: Pytree that precedes the (Hadamard) multiplication operator. 
  rest: Pytree that succeeds the (Hadamard) multiplication operator. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf (default ``None``). 

Returns:
  Resulting pytree. 
"""

tree_div = functools.partial(tree_map, operator.truediv)
tree_div.__doc__ = r"""
Compute tree division.

This function computes :math:`tree \oslash rest`.

Args: 
  tree: Pytree that precedes the (Hadamard) division operator. 
  rest: Pytree that succeeds the (Hadamard) division operator. 
  is_leaf: Optional function used to decide what component of the tree is considered a 
    leaf (default ``None``). 

Returns:
  Resulting pytree. 
"""


def tree_scalar_mul(scalar, tree_x):
    r"""Compute a tree multiplied by a scalar.

    The function computes :math:`scalar \cdot tree_x`.

    Args:
      scalar: Scalar to be applied to the tree.
      tree_x: Pytree to be multiplied by the scalar.

    Returns:
      Resulting pytree.
    """
    return tree_map(lambda x: scalar * x, tree_x)


def tree_add_scalar_mul(tree_x, scalar, tree_y):
    r"""Compute the sum of a tree and another scalar multiplied tree.

    The function computes :math:`tree_x + scalar \cdot tree_y`.

    Args:
      tree_x: Pytree to add to the scalar multiplied tree.
      scalar: Scalar to be applied to the tree.
      tree_y: Pytree to be multiplied by the scalar.

    Returns:
      Resulting pytree.
    """
    return tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)  # type: ignore


def _vdot_safe(a, b):
    return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_vdot(tree_x, tree_y):
    r"""Compute the inner product of trees.

    The function computes :math:`\langle tree_x, \, tree_y \rangle`.

    Args:
      tree_x: Pytree as the first argument to the inner product.
      tree_y: Pytree as the second argument to the inner product.

    Returns:
      Resulting inner product.
    """
    vdots = tree_map(_vdot_safe, tree_x, tree_y)
    return tree_reduce(operator.add, vdots)


def _vdot_real(x, y):
    r"""Vector dot-product guaranteed to have a real valued result despite possibly
    complex input. Thus neglects the real-imaginary cross-terms. The result is a real
    float.
    """
    # result = _vdot(x.real, y.real)
    # if jnp.iscomplexobj(x) and jnp.iscomplexobj(y):
    #  result += _vdot(x.imag, y.imag)
    result = _vdot(
        x, y
    ).real  # NOTE: without jit this is faster than variant above, no difference with jit
    return result


def tree_vdot_real(tree_x, tree_y):
    r"""Compute the real part of the inner product.

    The function computes :math:`\operatorname{Re}(\langle tree_x, tree_y \rangle)`.
    """
    return sum(tree_leaves(tree_map(_vdot_real, tree_x, tree_y)))


def tree_dot(tree_x, tree_y):
    r"""Compute leaves-wise dot product between pytree of arrays.

    The function computes :math:`\langle tree_x, tree_y \rangle' for pytree of arrays.

    This is useful to store block diagonal linear operators: each leaf of the tree
    corresponds to a block.
    """
    return tree_map(jnp.dot, tree_x, tree_y)


def tree_sum(tree_x):
    r"""Compute the sum of leaves of a tree.

    The function computes :math:`\sum_{leaf \in tree_x} leaf`.
    """
    sums = tree_map(jnp.sum, tree_x)
    return tree_reduce(operator.add, sums)


def tree_l2_norm(tree_x, squared=False):
    r"""Compute the :math:`\ell^2` norm of a tree.

    The function computes :math:`\lVert tree_x \rVert`. If ``squared`` is set to
    ``True``, it returns the squared norm instead.

    Args:
      tree_x: Pytree of interest.
      squared: Whether to compute the squared norm (default ``False``).

    Returns:
      Norm (or squared norm) of the tree.
    """
    squared_tree = tree_map(
        lambda leaf: jnp.square(leaf.real) + jnp.square(leaf.imag), tree_x
    )
    sqnorm = tree_sum(squared_tree)
    if squared:
        return sqnorm
    else:
        return jnp.sqrt(sqnorm)


def tree_zeros_like(tree_x):
    r"""Create an all-zero tree with the same structure.

    Args:
      tree_x: Pytree whose structure and data-type define the returned tree.

    Returns:
      Pytree of zeros with the same structure and type as ``tree_x``.
    """
    return tree_map(jnp.zeros_like, tree_x)


def tree_ones_like(tree_x):
    r"""Create an all-ones tree with the same structure.

    Args:
      tree_x: Pytree whose structure and data-type define the returned tree.

    Returns:
      Pytree of ones with the same structure and type as ``tree_x``.
    """
    return tree_map(jnp.ones_like, tree_x)


def tree_average(trees, weights):
    r"""Return the weighted linear combination of a list of trees.

    The function computes :math:`\sum_{i=1}^{\text{num\_trees}} weight_i \cdot tree_i`.

    Args:
      trees: Array of trees with shape ``(num_trees,...)``.
      weights: Array of weights with shape ``(num_trees,)``.

    Returns:
      A single tree that is the weighted linear combination of all the trees.
    """
    return tree_map(lambda x: jnp.tensordot(weights, x, axes=1), trees)


def tree_gram(trees):
    r"""Compute Gram matrix from pytrees.

    The function computes matrix :math:`G` given by :math:`G_{i,j} = \langle tree_i,
    tree_j \rangle`.

    Args:
      trees: Array of trees with shape ``(num_trees,...)``.

    Returns:
      Arrays of shape ``(num_trees, num_trees)`` of all dot products.
    """
    vmap_left = jax.vmap(tree_vdot, in_axes=(0, None))
    vmap_right = jax.vmap(vmap_left, in_axes=(None, 0))
    return vmap_right(trees, trees)


def tree_inf_norm(tree_x):
    r"""Compute the infinity norm of a pytree.

    The function computes :math:`\lVert tree_x \rVert_{\infty}`.
    """
    leaves_vec = tree_leaves(tree_map(jnp.ravel, tree_x))
    return jnp.max(jnp.abs(jnp.concatenate(leaves_vec)))


def tree_where(cond, a, b):
    r"""jnp.where for trees.

    Mimic broadcasting semantic of :func:`jax.numpy.where`.
    ``cond``, ``a`` and ``b`` can be arrays (including scalars) broadcastable to the
    leaves of the other input arguments.

    Args:
      cond: Pytree of booleans arrays, or single array broadcastable to the shapes of
        leaves of ``a`` and ``b``.
      a: Pytree of arrays, or single array broadcastable to the shapes of leaves of
        ``cond`` and ``b``.
      b: Pytree of arrays, or single array broadcastable to the shapes of leaves of
        ``cond`` and ``a``.

    Returns:
      Pytree of arrays, or single array
    """
    cond, a, b = broadcast_pytrees(cond, a, b)
    return tree_map(jnp.where, cond, a, b)


def tree_negative(tree):
    r"""Compute leaf-wise negation.

    The function computes :math:`-tree`.
    """
    return tree_scalar_mul(-1, tree)


def tree_reciprocal(tree):
    r"""Compute leaf-wise inverse.

    The function computes :math:`1 \oslash tree`. In other words, the function returns a
    pytree that consists of :math:`1/leaf` where :math:`leaf` is the corresponding leaf
    of the original tree :math:`tree`.
    """
    return tree_map(lambda x: jnp.reciprocal(x), tree)


def tree_mean(tree):
    r"""Mean reduction for a tree.

    The function computes :math:`\frac{1}{\text{num_leaves}} \sum_{leaf \in tree}
    \frac{1}{\lvert leaf \rvert} \sum_{i \in leaf} i`.
    """
    leaves_avg = tree_map(jnp.mean, tree)
    return tree_sum(leaves_avg) / len(tree_leaves(leaves_avg))


def tree_single_dtype(tree, convert_in_jax_dtype=True):
    r"""The dtype for all values in a tree, provided that all leaves share the same
    type.

    If the leaves have different type, raise a ``ValueError``.

    Args:
      tree: Tree to get the dtype of.
      convert_in_jax_type: Whether to convert the types in JAX precision.
        Namely, a ``numpy`` ``int64`` type is converted in a ``jax.numpy`` ``int32``
        type by default unless one enables double precision using
        ``jax.config.update("jax_enable_x64", True)``.

    Return:
      dtype shared by all leaves of the tree.
    """
    if convert_in_jax_dtype:
        dtypes = set(
            jnp.asarray(p).dtype
            for p in jtu.tree_leaves(tree)
            if isinstance(p, (bool, int, float, complex, np.ndarray, jnp.ndarray))
        )
    else:
        dtypes = set(
            np.asarray(p).dtype
            for p in jtu.tree_leaves(tree)
            if isinstance(p, (bool, int, float, complex, np.ndarray, jnp.ndarray))
        )
    if not dtypes:
        return None
    if len(dtypes) == 1:
        dtype = dtypes.pop()
        return dtype
    raise ValueError("Found more than one dtype in the tree.")


def get_real_dtype(dtype):
    r"""Dtype corresponding of real part of a complex dtype."""
    if dtype not in [f"complex{i}" for i in [4, 8, 16, 32, 64, 128]]:
        return dtype
    else:
        return dtype.type(0).real.dtype


def tree_conj(tree):
    r"""Complex conjugate of a tree.

    The function computes :math:`\overline{tree}` where conjugacy applies to the tree
    leaf-wise.
    """
    return tree_map(jnp.conj, tree)


def tree_real(tree):
    r"""Real part of a tree.

    The function computes :math:`\operatorname{Re}(tree)` where
    :math:`\operatorname{Re}` applies to the tree leaf-wise.
    """
    return tree_map(jnp.real, tree)


def tree_imag(tree):
    r"""Imaginary part of a tree.

    The function computes :math:`\operatorname{Im}(tree)` where
    :math:`\operatorname{Im}` applies to the tree leaf-wise.
    """
    return tree_map(jnp.imag, tree)
