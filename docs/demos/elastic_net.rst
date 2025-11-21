Elastic Net Regression
======================

This example demonstrates solving an Elastic Net regression problem, which combines L1 and L2 regularization:

.. math::

   \minimize_{\beta} \frac{1}{2}\|X\beta - y\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2

Problem Setup
-------------

.. code-block:: python

   import torch
   from rlaopt.expression import Variable
   from rlaopt.atoms import ElasticNet, SumSquares
   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Generate synthetic data
   n_samples, n_features = 100, 50
   X = torch.randn(n_samples, n_features)
   y = torch.randn(n_samples)

   # Create variables
   beta = Variable((n_features,), name='beta')
   X_var = Variable((n_samples, n_features), name='X')
   y_var = Variable((n_samples,), name='y')

   # Set data
   X_var.value = X
   y_var.value = y

Building the Objective
----------------------

We can use the ElasticNet atom directly:

.. code-block:: python

   # Regularization parameters
   lambda_l1 = 0.1
   lambda_l2 = 0.01

   # Build objective
   residual = X_var @ beta - y_var
   data_fit = SumSquares(residual)
   regularization = ElasticNet(beta, l1_scaling=lambda_l1, l2_scaling=lambda_l2)
   objective = data_fit + regularization

Solving
-------

.. code-block:: python

   config = ProxGradConfig(eta=0.01, use_linesearch=True, max_iters=1000, tol=1e-4)
   solver = ProxGrad(objective, config)
   result = solver.solve()

   print(f"Solution: {beta.value.data}")


