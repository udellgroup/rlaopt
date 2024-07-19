Base
====

.. currentmodule:: sketchyopts.base

.. autosummary::
    SolverState
    PromiseSolver
    LinearOperator
    HessianLinearOperator

PROMISE Solver
~~~~~~~~~~~~~~
.. autoclass:: SolverState
.. autoclass:: PromiseSolver

Linear Operator
~~~~~~~~~~~~~~~
.. autoclass:: LinearOperator
    :members:
    :special-members: __init__, __matmul__
.. autoclass:: HessianLinearOperator
    :members:
    :special-members: __init__