Operator
========

.. currentmodule:: sketchyopts.operator

.. autosummary::
    LinearOperator
    HessianLinearOperator
    AddLinearOperator

Base Linear Operator
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LinearOperator
    :members:
    :special-members: __init__, __matmul__

Hessian Operator
~~~~~~~~~~~~~~~~
.. autoclass:: HessianLinearOperator
    :members:
    :special-members: __init__

Operator Addition
~~~~~~~~~~~~~~~~~
.. autoclass:: AddLinearOperator
    :members:
    :special-members: __init__