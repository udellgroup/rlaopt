Basic Solver Usage
==================

This example provides a simple introduction to using rlaopt solvers.

Simple Quadratic Problem
------------------------

Let's solve a simple quadratic optimization problem:

.. code-block:: python

   import torch
   from rlaopt.expression import Variable, Constant
   from rlaopt.atoms import SumSquares
   from rlaopt.solvers import ProxGrad, ProxGradConfig

   # Create a variable
   x = Variable((5,), name='x')

   # Build a simple objective: ||x - target||^2
   target = torch.ones(5)
   # target is data, not a variable
   target_const = Constant(target)

   objective = SumSquares(x - target_const)

   # Solve
   config = ProxGradConfig(eta=0.1, max_iters=100, tol=1e-6)
   solver = ProxGrad(objective, config)
   result = solver.solve()

   print(f"Solution: {result.variable_values}")
   print(f"Target: {target}")
   print(f"Final error: {result.err}")
   print(f"Distance to target: {torch.linalg.norm(result.variable_values.to_flat_tensor() - target)}")

This should find :math:`x = \text{target}`, minimizing the squared distance.
