Utilities
=========

.. currentmodule:: sketchyopts.util

.. autosummary::
    LinearOperator
    shareble_state_named_chain
    scale_by_ref_learning_rate

Linear Operator
~~~~~~~~~~~~~~~
.. autoclass:: LinearOperator
    :members:
    :special-members: __init__, __matmul__

Transformations
~~~~~~~~~~~~~~~
.. autofunction:: shareble_state_named_chain
.. autofunction:: scale_by_ref_learning_rate