Lasso Regression
=================

This example demonstrates how to solve a Lasso regression problem using rlaopt. Lasso regression solves:

.. math::

   \minimize_{\beta} \frac{1}{2}\|X\beta - y\|_2^2 + \lambda \|\beta\|_1

where :math:`X` is the design matrix, :math:`y` is the target vector, and :math:`\lambda` is the regularization parameter.

Problem Setup
-------------

First, we'll generate some synthetic data:

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt

   # Set random seed for reproducibility
   torch.manual_seed(42)
   np.random.seed(42)

   # Generate synthetic data
   n_samples = 100
   n_features = 50
   n_nonzero = 10  # Number of non-zero coefficients in true solution

   # True sparse solution
   beta_true = torch.zeros(n_features)
   beta_true[:n_nonzero] = torch.randn(n_nonzero)

   # Generate design matrix
   X = torch.randn(n_samples, n_features)
   X = X / torch.norm(X, dim=0)  # Normalize columns

   # Generate target with noise
   y = X @ beta_true + 0.1 * torch.randn(n_samples)

Building the Problem
--------------------

Now we'll set up the optimization problem using rlaopt:

.. code-block:: python

   from rlaopt.expression import Variable, Constant
   from rlaopt.atoms import L1Norm, SumSquares
   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Create optimization variable
   beta = Variable((n_features,), name='beta')

   # X and y are data tensors, not variables
   X_const = Constant(X)
   y_const = Constant(y)

   # Regularization parameter
   lambda_reg = 0.1

   # Build the objective: ||X*beta - y||^2 + lambda * ||beta||_1
   residual = X_const @ beta - y_const
   data_fit = SumSquares(residual)
   regularization = L1Norm(beta, scaling=lambda_reg)
   objective = data_fit + regularization

Solving the Problem
-------------------

Configure and run the solver:

.. code-block:: python

   # Configure the solver
   config = ProxGradConfig(
       eta=0.01,              # Initial step size
       use_linesearch=True,   # Use backtracking line search
       use_acceleration=False, # Don't use Nesterov acceleration
       max_iters=1000,        # Maximum iterations
       tol=1e-4              # Convergence tolerance
   )

   # Create and run the solver
   solver = ProxGrad(objective, config)
   # solver.solve() returns (variable_values, final_error)
   variable_values, final_error = solver.solve()

   # Get the solution
   beta_opt = beta.value.data

Results
-------

Let's compare the solution with the true coefficients:

.. code-block:: python

   # Compute metrics
   mse = torch.mean((beta_opt - beta_true) ** 2)
   n_nonzero_opt = (torch.abs(beta_opt) > 1e-3).sum().item()
   n_nonzero_true = (torch.abs(beta_true) > 1e-3).sum().item()

   print(f"Mean squared error: {mse:.4f}")
   print(f"True non-zero coefficients: {n_nonzero_true}")
   print(f"Estimated non-zero coefficients: {n_nonzero_opt}")

   # Visualize the solution
   plt.figure(figsize=(12, 5))

   plt.subplot(1, 2, 1)
   plt.plot(beta_true.numpy(), 'o-', label='True', alpha=0.7)
   plt.plot(beta_opt.numpy(), 's-', label='Estimated', alpha=0.7)
   plt.xlabel('Feature index')
   plt.ylabel('Coefficient value')
   plt.title('Coefficient Comparison')
   plt.legend()
   plt.grid(True, alpha=0.3)

   plt.subplot(1, 2, 2)
   plt.scatter(beta_true.numpy(), beta_opt.numpy(), alpha=0.6)
   plt.plot([beta_true.min(), beta_true.max()], 
            [beta_true.min(), beta_true.max()], 'r--', alpha=0.5)
   plt.xlabel('True coefficients')
   plt.ylabel('Estimated coefficients')
   plt.title('Coefficient Scatter Plot')
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Complete Code
-------------

Here's the complete example in one block:

.. literalinclude:: ../../examples/lasso_example.py
   :language: python
   :linenos:

Notes
-----

* The Lasso problem is well-suited for the ProxGrad solver because:
  * The data-fitting term (sum of squares) is smooth
  * The L1 regularization term is proxable
* The line search helps adapt the step size automatically
* The regularization parameter :math:`\lambda` controls the sparsity of the solution
* Larger :math:`\lambda` values lead to sparser solutions



