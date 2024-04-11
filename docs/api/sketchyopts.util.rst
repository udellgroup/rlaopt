Utilities
=========

.. currentmodule:: sketchyopts.util

.. autosummary::
    LinearOperator
    HessianLinearOperator
    TransformUpdateExtraArgsRefStateFn
    GradientTransformationExtraArgsRefState
    with_ref_state_support
    shareble_state_named_chain
    scale_by_ref_learning_rate

Linear Operator
~~~~~~~~~~~~~~~
.. autoclass:: LinearOperator
    :members:
    :special-members: __init__, __matmul__
.. autoclass:: HessianLinearOperator
    :members:
    :special-members: __init__

Transformations
~~~~~~~~~~~~~~~

Type
^^^^
.. autoclass:: TransformUpdateExtraArgsRefStateFn
    :special-members: __call__
.. autoclass:: GradientTransformationExtraArgsRefState

Type Conversion
^^^^^^^^^^^^^^^
.. autofunction:: with_ref_state_support

Sharable State Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: shareble_state_named_chain
.. autofunction:: scale_by_ref_learning_rate