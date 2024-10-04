Proximal Operators
==================

.. currentmodule:: sketchyopts.prox

.. autosummary::
    prox_const
    prox_l1
    prox_nonnegative_l1
    prox_elastic_net
    prox_l2
    prox_l2_squared
    prox_nonnegative_l2_squared
    prox_nonnegative
    prox_box
    prox_hyperplane
    prox_halfspace

.. note:: 
    Implementation of the following proximal operators is adapted from JAXopt 
    `proximal operators 
    <https://jaxopt.github.io/stable/non_smooth.html#proximal-operators>`_ and 
    `projections 
    <https://jaxopt.github.io/stable/constrained.html#projections>`_. Original code is 
    available from the official repo: `source 1 
    <https://github.com/google/jaxopt/blob/main/jaxopt/_src/prox.py>`_ and `source 2 
    <https://github.com/google/jaxopt/blob/main/jaxopt/_src/projection.py>`_.


Constant Function
~~~~~~~~~~~~~~~~~
.. autofunction:: prox_const

Norms
~~~~~
.. autofunction:: prox_l1
.. autofunction:: prox_nonnegative_l1
.. autofunction:: prox_elastic_net
.. autofunction:: prox_l2
.. autofunction:: prox_l2_squared
.. autofunction:: prox_nonnegative_l2_squared

Indicators (projections)
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: prox_nonnegative
.. autofunction:: prox_box
.. autofunction:: prox_hyperplane
.. autofunction:: prox_halfspace