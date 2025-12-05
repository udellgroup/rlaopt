Solvers Module
===============

.. automodule:: rlaopt.solvers

Convergence Status
------------------

.. autoclass:: rlaopt.solvers.ConvergenceStatus
   :members:
   :undoc-members:

Alternating Direction Method of Multipliers
-------------------------------------------

.. autoclass:: rlaopt.solvers.ADMM
   :members:

.. autopydantic_model:: rlaopt.solvers.ADMMConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.ADMMStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.ADMMResult

Proximal Gradient
-----------------

.. autoclass:: rlaopt.solvers.ProxGrad
   :members:

.. autopydantic_model:: rlaopt.solvers.ProxGradConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.ProxGradStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.ProxGradResult

Preconditioned Conjugate Gradient
---------------------------------

.. autoclass:: rlaopt.solvers.PCG
   :members:

.. autopydantic_model:: rlaopt.solvers.PCGConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.PCGStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.PCGResult
