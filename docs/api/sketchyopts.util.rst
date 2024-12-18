Utilities
=========

.. currentmodule:: sketchyopts.util

.. autosummary::
    ravel_tree
    tree_map
    tree_flatten
    tree_unflatten
    tree_add
    tree_sub
    tree_scalar_mul
    tree_add_scalar_mul
    tree_vdot
    tree_l2_norm
    tree_zeros_like
    tree_ones_like
    default_floating_dtype
    default_integer_dtype
    inexact_asarray
    integer_asarray
    is_array
    frozenset
    sample_indices
    form_dense_vector

.. currentmodule:: sketchyopts.sketching

.. autosummary::
    hadamard_transform

.. currentmodule:: sketchyopts.util

Pytree Utilities
~~~~~~~~~~~~~~~~

.. note:: 
    Implementation of the following pytree utility functions is adapted from `JAXopt 
    <https://jaxopt.github.io/stable/api.html#tree-utilities>`_. Original code is 
    available from the `official repo 
    <https://github.com/google/jaxopt/blob/main/jaxopt/_src/tree_util.py>`_. 

.. autofunction:: ravel_tree
.. autofunction:: tree_map
.. autofunction:: tree_flatten
.. autofunction:: tree_unflatten
.. autofunction:: tree_add
.. autofunction:: tree_sub
.. autofunction:: tree_scalar_mul
.. autofunction:: tree_add_scalar_mul
.. autofunction:: tree_vdot
.. autofunction:: tree_l2_norm
.. autofunction:: tree_zeros_like
.. autofunction:: tree_ones_like

Data Type Utilities
~~~~~~~~~~~~~~~~~~~

.. autofunction:: default_floating_dtype
.. autofunction:: default_integer_dtype
.. autofunction:: inexact_asarray
.. autofunction:: integer_asarray
.. autofunction:: is_array
.. autofunction:: frozenset

Sketching Utilities
~~~~~~~~~~~~~~~~~~~

.. autofunction:: sample_indices
.. autofunction:: form_dense_vector
.. autofunction:: sketchyopts.sketching.hadamard_transform
