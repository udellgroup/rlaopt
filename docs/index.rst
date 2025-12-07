.. rlaopt documentation master file

Welcome to rlaopt
==================

**rlaopt** is a Python package for randomized linear algebra-based optimization algorithms in PyTorch. It provides a flexible framework for building and solving optimization problems using symbolic expressions, atoms, and efficient solvers.

.. note::
   This package is under active development. The API may change frequently, and the code may not be stable. Use at your own risk.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

Basic Usage
~~~~~~~~~~~

Here's a simple example of creating variables and building expressions:

.. code-block:: python

   import torch
   from rlaopt.expression import Variable, Constant
   from rlaopt.atoms import L1Norm, SumSquares
   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Create a variable
   x = Variable((10,), name='x')

   # Build an objective: ||Ax - b||^2 + lambda * ||x||_1
   # A and b are data matrices
   A = Constant(torch.randn(5, 10))
   b = Constant(torch.randn(5))
   lambda_reg = 0.1

   # Create the objective expression
   residual = A @ x - b
   objective = SumSquares(residual) + L1Norm(x, scaling=lambda_reg)

   # Solve using proximal gradient
   config = ProxGradConfig(eta=0.01, use_linesearch=True)
   solver = ProxGrad(objective, config)
   result = solver.solve()

Lasso Regression Example
~~~~~~~~~~~~~~~~~~~~~~~~

A complete example of solving a Lasso regression problem:

.. code-block:: python

   import torch
   from rlaopt.expression import Variable
   from rlaopt.atoms import L1Norm, SumSquares
   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Generate synthetic data
   n_samples, n_features = 100, 50
   X = torch.randn(n_samples, n_features)
   y = torch.randn(n_samples)
   lambda_reg = 0.1

   # Create optimization variable
   beta = Variable((n_features,), name='beta')

   # X and y are data tensors
   from rlaopt.expression import Constant
   X_const = Constant(X)
   y_const = Constant(y)

   # Build Lasso objective: ||X*beta - y||^2 + lambda * ||beta||_1
   residual = X_const @ beta - y_const
   objective = SumSquares(residual) + L1Norm(beta, scaling=lambda_reg)

   # Solve
   config = ProxGradConfig(eta=0.01, max_iters=1000, tol=1e-4)
   solver = ProxGrad(objective, config)
   result = solver.solve()

   # Update variables with the solution
   objective.update_variables(result.solution)

   print(f"Solution: {beta.value}")
   print(f"Final error: {result.err}")

Key Features
------------

* **Symbolic Expressions**: Build optimization problems using natural mathematical syntax
* **Atoms**: Pre-built optimization atoms (L1 norm, sum of squares, elastic net, etc.)
* **Efficient Solvers**: Proximal gradient, PCG, and other randomized linear algebra solvers
* **PyTorch Integration**: Seamless integration with PyTorch's autograd and device management
* **Flexible**: Support for both smooth and non-smooth optimization problems

Documentation Structure
------------------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation
   key_concepts/index
   demos/index
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
