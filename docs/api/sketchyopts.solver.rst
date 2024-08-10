.. role:: python(code)
  :language: python
  :class: highlight

Solvers
=======

.. currentmodule:: sketchyopts.solver

.. autosummary::
    nystrom_pcg
    SketchySGD
    SketchySGDState
    SketchySVRG
    SketchySVRGState
    SketchySAGA
    SketchySAGAState
    SketchyKatyusha
    SketchyKatyushaState

Nyström PCG
~~~~~~~~~~~
.. autofunction:: nystrom_pcg

PROMISE Solvers
~~~~~~~~~~~~~~~
SketchyOpts includes four sketching-based preconditioned stochastic gradient algorithms (PROMISE solvers): :ref:`api/sketchyopts.solver:SketchySGD`, :ref:`api/sketchyopts.solver:SketchySVRG`, :ref:`api/sketchyopts.solver:SketchySAGA`, and :ref:`api/sketchyopts.solver:SketchyKatyusha`. These solvers share some common arguments and we provide detailed guidance on a few of them here. Please go to each individual section below for solver-specific description and a complete list of its arguments. 

.. glossary::

    ``fun``
        ``fun`` specifies the objective function of the optimization problem. The argument is **required**. It must have: 
        
        - scalar output
        - optimization variable as **first** argument
        - argument ``data`` for data input, and argument ``reg`` for :math:`\ell^2` regularization strength

        For instance, ``fun`` could take the form :python:`value = fun(params, some_arg, data, reg, other_arg)`. 

        Please note that :class:`SketchySAGA` has additional requirement for the objective function ``fun``. Please go to the section to find out more. 

----

.. glossary::

    ``grad_fun`` 
        ``grad_fun`` specifies the gradient oracle that computes the gradient of the objective function ``fun`` with respect to its first argument (*i.e.* the optimization variable). The argument is **optional**. 

        - It is expected to have the same function signature as the objective function ``fun``. For instance, ``grad_fun`` could take the form :python:`grad = grad_fun(params, some_arg, data, reg, other_arg)`.
        - The gradient output should have the same shape and type as the the optimization variable. 

----

.. glossary::
    
    ``precond`` 
        ``precond`` specifies the type of preconditioner for the solver. Two types of preconditioner are available: Nyström subsampled Newton (:python:`precond = nyssn`) by default, and subsampled Newton (:python:`precond = ssn`). Each type relies on a different oracle. 

        - If Nyström subsampled Newton is selected, the Hessian-vector product oracle ``hvp_fun`` can be **optionally** provided.
        - If subsampled Newton is selected, the square-root Hessian oracle ``sqrt_hess_fun`` is **required**. 

----

.. glossary::

    ``hvp_fun``
        ``hvp_fun`` specifies the Hessian-vector product oracle that computes the product of the Hessian and an arbitrary compatible vector. It is **optional** for using the Nyström subsampled Newton preconditioner. If provided, it must have: 

        - optimization variable as **first** argument
        - arbitrary vector (of the same shape and type as the optimization variable) as **second** argument
        - same additional arguments as the objective function ``fun`` (``data``, ``reg``, etc.)
        - function output of the same shape and type as the optimization variable

        For instance, ``hvp_fun`` could take the form :python:`vec_output = hvp_fun(params, vec, some_arg, data, reg, other_arg)`. 

----

.. glossary::

    ``sqrt_hess_fun``
        ``sqrt_hess_fun`` specifies the square-root Hessian oracle. It is **mandatory** for using the subsampled Newton preconditioner. 

        - It is expected to return a **2-dimensional** array :math:`X` with :math:`X^{T}X = H` where :math:`H` is the Hessian of the **unregularized** objective function (*i.e.* :math:`\ell^2` regularization strength set to :python:`reg = 0`) with respect to the **flattened** first argument. 
        - It should take the same arguments as the objective function ``fun`` other than the regularization strength ``reg``. For instance, ``sqrt_hess_fun`` could take the form :python:`matrix_output = sqrt_hess_fun(params, some_arg, data, other_arg)`.

----

.. note::
    It is unnecessary to JIT-compile the objective function ``fun`` or any of the above oracles (``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun``) beforehand because the PROMISE solver will apply JIT transformation and perform JIT-compilation internally when initialized with :python:`jit = True`.

----

.. glossary::
    
    ``pre_update``
        ``pre_update`` specifies the **optional** callback function that gets called first in each iteration before the solver updates the iterate. 

        - The function expects signature :python:`params, state = pre_update(params, state, *args, **kwargs, data, reg)` where ``state`` is the corresponding solver state object (*i.e.* :class:`SketchySGDState`, :class:`SketchySVRGState`, :class:`SketchySAGAState`, or :class:`SketchyKatyushaState`). 
        - The additional arguments :python:`*args` and :python:`**kwargs` have to be consistent with the objective function ``fun`` (meaning they all are arguments accepted by ``fun`` as well). 

        With our running example, ``pre_update`` could take the form :python:`pre_update(params, state, some_arg, data, reg, other_arg)`. 

.. note::
    To accommodate various use cases, SketchyOpts does not require ``pre_update`` to be JIT-compilable. This flexibility means :python:`jit = True` has no effect on ``pre_update``. The function needs be JIT transformed before getting passed to the PROMISE solver if user wishes to JIT-compile the function for faster execution. 

----

.. glossary::

    ``learning_rate``
        ``learning_rate`` specifies the scaling of each update to the optimization variable. PROMISE solvers accept either a fixed scalar or a callable. 

        - If a fixed scalar value is provided, the solver uses the value as the multiplier to the adaptive learning rate obtained from preconditioner update. 
        - If a callable function is provided, the solver views it as a schedule and no longer computes adaptive learning rate when updating the preconditioner. 

        Please note that :class:`SketchyKatyusha` does not use ``learning_rate`` as it is momentum accelerated. 
        
----

.. glossary::
    
    ``jit``
        ``jit`` specifies whether to enable JIT-compilation in the PRIMISE solver. If :python:`jit = True` (by default), the objective function ``fun`` and all of the provided oracles (``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun``) must be compatible with JIT-compilation. 

.. seealso::
    For further details on how JIT-compilation works in JAX as well as useful tips on making function JIT-compilable, please see the `JAX Just-in-time compilation tutorial <https://jax.readthedocs.io/en/latest/jit-compilation.html>`_.

----

.. glossary::
    
    ``sparse``
        ``sparse`` specifies whether to enable sparse inputs support in the PRIMISE solver. If :python:`sparse = True` (default ``False``), the PROMISE solver will sparsify its internal components; thus the objective function ``fun``, all of the provided oracles (``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun``), and the callback ``pre_update`` must all support sparse inputs in this case. 

.. caution::
    The support for sparse matrix operations in JAX is currently under development, and is considered experimental. As a result, this sparsification feature of PROMISE solvers is preliminary and might not be performant at the moment. We plan to improve this feature in future versions of SketchyOpts. 


SketchySGD
----------
.. autoclass:: SketchySGD
    :special-members: run
.. autoclass:: SketchySGDState

SketchySVRG
-----------
.. autoclass:: SketchySVRG
    :special-members: run
.. autoclass:: SketchySVRGState

SketchySAGA
-----------
.. autoclass:: SketchySAGA
    :special-members: run
.. autoclass:: SketchySAGAState

SketchyKatyusha
---------------
.. autoclass:: SketchyKatyusha
    :special-members: run
.. autoclass:: SketchyKatyushaState