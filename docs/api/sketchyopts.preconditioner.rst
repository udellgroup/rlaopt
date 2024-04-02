Preconditioners
===============

.. currentmodule:: sketchyopts.preconditioner

.. autosummary::
    rand_nystrom_approx
    NystromPrecondState
    update_nystrom_precond

Randomized Nyström Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rand_nystrom_approx

Nyström Preconditioner
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: NystromPrecondState
    :members:
.. autofunction:: update_nystrom_precond
.. autofunction:: scale_by_nystrom_precond