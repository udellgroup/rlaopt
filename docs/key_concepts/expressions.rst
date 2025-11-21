Expressions
===========

Expressions are the fundamental building blocks of optimization problems in rlaopt. They represent mathematical operations that can be evaluated, differentiated, and optimized.

Core Expression Types
---------------------

Variable
~~~~~~~~

A :class:`~rlaopt.expression.variable.Variable` represents an optimization variable - a parameter that will be optimized. Variables wrap PyTorch parameters and support automatic differentiation.

.. code-block:: python

   from rlaopt.expression import Variable

   # Create a variable with shape (10,)
   x = Variable((10,), name='x')

   # Create a matrix variable
   A = Variable((5, 10), name='A')

Constant
~~~~~~~~

A :class:`~rlaopt.expression.constant.Constant` represents a fixed value that doesn't change during optimization.

.. code-block:: python

   from rlaopt.expression import Constant
   import torch

   # Create a constant from a tensor
   c = Constant(torch.ones(5))

Building Expressions
--------------------

Expressions can be combined using standard mathematical operations:

.. code-block:: python

   from rlaopt.expression import Variable

   x = Variable((10,), name='x')
   A = Variable((5, 10), name='A')
   b = Variable((5,), name='b')

   # Linear combination
   y = A @ x + b

   # Element-wise operations
   z = x ** 2
   w = x + 1.0

   # Composition
   expr = (A @ x - b) ** 2

Expression Properties
---------------------

All expressions have several important properties:

* :meth:`~rlaopt.expression.expression.Expression.is_smooth`: Whether the expression is differentiable everywhere
* :meth:`~rlaopt.expression.expression.Expression.forward`: Evaluate the expression
* :meth:`~rlaopt.expression.expression.Expression.tree`: Get a tree representation of the expression structure

.. code-block:: python

   # Check if expression is smooth
   is_smooth = expr.is_smooth()

   # Evaluate the expression
   result = expr.forward()

   # Get expression tree
   tree = expr.tree()

Operator Overloading
--------------------

rlaopt supports natural mathematical syntax through operator overloading:

* ``+``, ``-``: Addition and subtraction
* ``*``: Element-wise multiplication
* ``@``: Matrix multiplication
* ``/``: Division
* ``**``: Exponentiation

This allows you to write optimization problems in a natural, readable way.


