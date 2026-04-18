Atoms Module
=============

.. automodule:: rlaopt.atoms

Base Classes
------------

.. autoclass:: rlaopt.atoms.Atom
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: prox, get_input, decompose

Regularizers
------------

.. autoclass:: rlaopt.atoms.L1Norm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.L2Norm
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.SumSquares
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.ElasticNet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.NucNorm
   :members:
   :undoc-members:
   :show-inheritance:

Constraints
-----------

.. autoclass:: rlaopt.atoms.Box
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.NonNegative
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.LinearEquality
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.Halfspace
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.Polyhedron
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: decompose

Regression Models
-----------------

.. autoclass:: rlaopt.atoms.LinearRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.LogisticRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.PoissonRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.GammaRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.InverseGaussianRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.CompoundPoissonGammaRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.HuberRegression
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rlaopt.atoms.MultinomialRegression
   :members:
   :undoc-members:
   :show-inheritance:
