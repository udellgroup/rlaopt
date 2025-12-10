Atoms
=====

Atoms are pre-built mathematical functions that can be used in optimization objectives and constraints. They extend the base :class:`~rlaopt.atoms.atom.Atom` class and provide efficient implementations of common optimization functions.

What are Atoms?
---------------

Atoms represent mathematical functions with specific properties:

* **Smoothness**: Whether the function is differentiable everywhere
* **Proxability**: Whether the function has an efficient proximal operator

These properties determine which optimization algorithms can be used with the atom.

Available Atoms
---------------

L1 Norm
~~~~~~~

The :class:`~rlaopt.atoms.L1Norm` atom represents the L1 (Manhattan) norm: :math:`\|x\|_1 = \sum_i |x_i|`.

.. code-block:: python

   from rlaopt.atoms import L1Norm
   from rlaopt.expression import Variable

   x = Variable((10,), name='x')
   l1 = L1Norm(x, scaling=0.1)  # 0.1 * ||x||_1

L2 Norm
~~~~~~~

The :class:`~rlaopt.atoms.L2Norm` atom represents the L2 (Euclidean) norm: :math:`\|x\|_2 = \sqrt{\sum_i x_i^2}`.

.. code-block:: python

   from rlaopt.atoms import L2Norm
   from rlaopt.expression import Variable

   x = Variable((10,), name='x')
   l2 = L2Norm(x, scaling=0.1)  # 0.1 * ||x||_2

Sum of Squares
~~~~~~~~~~~~~~

The :class:`~rlaopt.atoms.SumSquares` atom represents the squared L2 norm: :math:`\|x\|_2^2 = \sum_i x_i^2`.

.. code-block:: python

   from rlaopt.atoms import SumSquares
   from rlaopt.expression import Variable

   x = Variable((10,), name='x')
   sos = SumSquares(x)

Elastic Net
~~~~~~~~~~~

The :class:`~rlaopt.atoms.ElasticNet` atom combines L1 and L2 regularization:
:math:`\lambda_1 \|x\|_1 + \lambda_2 \|x\|_2^2`.

.. code-block:: python

   from rlaopt.atoms import ElasticNet
   from rlaopt.expression import Variable

   x = Variable((10,), name='x')
   elastic = ElasticNet(x, l1_scaling=0.1, l2_scaling=0.01)

Other Atoms
~~~~~~~~~~~

rlaopt provides many other atoms:

* :class:`~rlaopt.atoms.L2Norm`: L2 (Euclidean) norm
* :class:`~rlaopt.atoms.Box`: Box constraints
* :class:`~rlaopt.atoms.NonNegative`: Non-negativity constraints
* :class:`~rlaopt.atoms.LinearEquality`: Linear equality constraints
* :class:`~rlaopt.atoms.Halfspace`: Halfspace constraints
* :class:`~rlaopt.atoms.Polyhedron`: Polyhedral constraints
* :class:`~rlaopt.atoms.NucNorm`: Nuclear norm

See the :doc:`../api/atoms` section for a complete list.

Using Atoms in Objectives
-------------------------

Atoms can be combined with expressions to build optimization objectives:

.. code-block:: python

   from rlaopt.expression import Variable
   from rlaopt.atoms import L1Norm, SumSquares

   x = Variable((10,), name='x')
   A = Variable((5, 10), name='A')
   b = Variable((5,), name='b')

   # Build objective: ||Ax - b||^2 + lambda * ||x||_1
   residual = A @ x - b
   objective = SumSquares(residual) + L1Norm(x, scaling=0.1)

Atom Properties
---------------

Each atom provides information about its properties:

.. code-block:: python

   atom = L1Norm(x)

   # Check if smooth
   print(atom.is_smooth())  # False

   # Check if proxable
   print(atom.is_proxable())  # True

These properties help determine which solvers can be used with the atom.


