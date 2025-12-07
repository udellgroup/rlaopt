Optimization Workflow
=====================

This page provides a step-by-step guide to building and solving optimization problems with rlaopt.

Step 1: Define Variables
-------------------------

First, create the optimization variables and data constants:

.. code-block:: python

   import torch
   from rlaopt.expression import Variable, Constant

   # Create optimization variable
   x = Variable((n_features,), name='x')

   # A and b are data matrices
   A = Constant(your_data_matrix)
   b = Constant(your_target_vector)

Step 2: Build the Objective
----------------------------

Construct your objective function using expressions and atoms:

.. code-block:: python

   from rlaopt.atoms import L1Norm, SumSquares

   # Build the objective expression
   residual = A @ x - b
   data_fit = SumSquares(residual)
   regularization = L1Norm(x, scaling=lambda_reg)
   objective = data_fit + regularization

Step 3: Choose a Solver
-----------------------

Select an appropriate solver based on your problem structure:

.. code-block:: python

   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Configure the solver
   config = ProxGradConfig(
       eta=0.01,
       use_linesearch=True,
       max_iters=1000,
       tol=1e-4
   )

   # Create the solver
   solver = ProxGrad(objective, config)

Step 4: Solve
-------------

Call the solver to find the solution:

.. code-block:: python

   # Solve the problem
   result = solver.solve()

   # Access the solution
   solution, err = result.variable_values, result.err
   print(f"Solution: {solution}")
   print(f"Final error: {err}")

Complete Example
----------------

Here's a complete example putting it all together:

.. code-block:: python

   import torch
   from rlaopt.expression import Variable, Constant
   from rlaopt.atoms import L1Norm, SumSquares
   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Step 1: Create variables and data
   n_samples, n_features = 100, 50
   x = Variable((n_features,), name='beta')

   # X and y are data tensors, not variables
   X = torch.randn(n_samples, n_features)
   y = torch.randn(n_samples)
   X_const = Constant(X)
   y_const = Constant(y)

   # Step 2: Build objective
   residual = X_const @ x - y_const
   objective = SumSquares(residual) + L1Norm(x, scaling=0.1)

   # Step 3 & 4: Solve
   config = ProxGradConfig(eta=0.01, max_iters=1000, tol=1e-4)
   solver = ProxGrad(objective, config)
   result = solver.solve()

   print(f"Optimal solution: {result.variable_values}")
   print(f"Final error: {result.err}")

Tips
----

* **Start simple**: Begin with basic examples and gradually add complexity
* **Check smoothness**: Use ``is_smooth()`` to understand which solvers work
* **Tune parameters**: Adjust step sizes and tolerances based on your problem
* **Monitor convergence**: Check solver state for convergence information
