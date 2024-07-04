import jax
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
