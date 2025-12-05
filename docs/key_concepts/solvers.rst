Solvers
=======

Solvers are algorithms that find solutions to optimization problems. rlaopt provides several solvers, each suited for different types of problems.

Available Solvers
-----------------

Alternating Direction Method of Multipliers (ADMM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~rlaopt.solvers.ADMM` solver is designed for problems of the form

.. math::

   \operatorname{minimize}_x f(x) + \sum_{i = 1}^k g_i(A_i x - b_i),

where :math:`f` is smooth (differentiable) and :math:`g` is proxable.

Our implementation internally decomposes the problem into the form

.. math::

   \operatorname{minimize}_{x, z_1, \ldots, z_k} f(x) + \sum_{i = 1}^k g_i(z_i) \\
   \text{subject to } A_i x - z_i - b_i = 0, \quad i = 1, \ldots, k

before applying the ADMM algorithm.

Our implementation is based on the implementation from :cite:t:`diamandis2023genios`.

.. code-block:: python

   from rlaopt.solvers import ADMM, ADMMConfig

   config = ADMMConfig(
       rho=1.0,               # Augmented Lagrangian parameter
   )
   solver = ADMM(objective, config)
   result = solver.solve()

Features:

* Supports multiple proxable terms for the same variable
* Allows for over-relaxation and primal-dual residual balancing :cite:p:`boyd2011distributed`

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
   result = solver.solve()

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
   result = solver.solve()

Choosing a Solver
-----------------

**Use ADMM when:**

* Your objective can be written in the form :math:`f(x) + \sum_{i = 1}^k g_i(A_i x - b_i)` where :math:`f` is smooth and :math:`g_i` are proxable
* If there are no non-smooth components, use ProxGrad instead

**Use ProxGrad when:**

* Your objective can be written in the form :math:`f(x) + g(x)` where :math:`f` is smooth and :math:`g` is proxable
* Examples include Lasso regression, Elastic Net, and other regularized regression problems

**Use PCG when:**

* You need to solve large positive-definite linear systems

Solver Configuration
--------------------

Each solver has a configuration class that controls its behavior:

* :class:`~rlaopt.solvers.ADMMConfig`: Augmented Lagrangian parameter, over-relaxation, residual balancing, etc.
* :class:`~rlaopt.solvers.ProxGradConfig`: Step size, line search, acceleration
* :class:`~rlaopt.solvers.PCGConfig`: Iteration limits, tolerance, preconditioner

Stopping Criteria
-----------------

All solvers support stopping criteria:

* :class:`~rlaopt.solvers.ADMMStoppingCriteria`: Maximum iterations, Primal and dual residual tolerance
* :class:`~rlaopt.solvers.ProxGradStoppingCriteria`: Maximum iterations, convergence tolerance
* :class:`~rlaopt.solvers.PCGStoppingCriteria`: Maximum iterations, convergence tolerance

The solver will stop when either criterion is met.

.. bibliography::
