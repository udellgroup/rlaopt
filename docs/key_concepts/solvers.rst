Solvers
=======

Solvers are algorithms that find solutions to optimization problems. rlaopt provides several solvers, each suited for different types of problems.

Available Solvers
-----------------

Proximal Gradient (ProxGrad)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~rlaopt.solvers.ProxGrad` solver is designed for problems of the form:

.. math::

   \operatorname{minimize}_x f(x) + g(x)

where :math:`f` is smooth (differentiable) and :math:`g` is proxable (has an efficient proximal operator).

.. code-block:: python

   from rlaopt.solvers import ProxGrad, ProxGradConfig

   config = ProxGradConfig(
       eta=0.01,              # Step size
       use_linesearch=True,   # Use line search
       use_acceleration=False # Use Nesterov acceleration
   )
   solver = ProxGrad(objective, config)
   variable_values, final_error = solver.solve()

Features:

* Supports backtracking line search
* Optional Nesterov acceleration

PCG (Preconditioned Conjugate Gradient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~rlaopt.solvers.PCG` solver is a preconditioned conjugate gradient method for solving linear systems.

.. code-block:: python

   from rlaopt.solvers import PCG, PCGConfig

   config = PCGConfig(max_iters=1000, tol=1e-6)
   solver = PCG(linear_system, config)
   solution, final_residual = solver.solve()

Choosing a Solver
-----------------

**Use ProxGrad when:**

* Your objective has both smooth and/or non-smooth components
* You have L1 regularization or other proxable penalties
* You need to solve problems like Lasso, Elastic Net, etc.

**Use PCG when:**

* You need to solve large positive-definite linear systems

Solver Configuration
--------------------

Each solver has a configuration class that controls its behavior:

* :class:`~rlaopt.solvers.ProxGradConfig`: Step size, line search, acceleration
* :class:`~rlaopt.solvers.PCGConfig`: Iteration limits, tolerance, preconditioner

.. code-block:: python

   config = ProxGradConfig(
       eta=0.01,
       use_linesearch=True,
       max_iters=1000,
       tol=1e-4
   )

Stopping Criteria
-----------------

All solvers support stopping criteria:

* ``max_iters``: Maximum number of iterations
* ``tol``: Convergence tolerance

The solver will stop when either criterion is met.


