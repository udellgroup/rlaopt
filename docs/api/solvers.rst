Solvers Module
===============

.. automodule:: rlaopt.solvers

Alternating Direction Method of Multipliers
-------------------------------------------

.. autoclass:: rlaopt.solvers.ADMM
   :members:

.. autopydantic_model:: rlaopt.solvers.ADMMConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.ADMMStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.ADMMResult
   :exclude-members: __init__

Proximal Gradient
-----------------

.. autoclass:: rlaopt.solvers.ProxGrad
   :members:

.. autopydantic_model:: rlaopt.solvers.ProxGradConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.ProxGradStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.ProxGradResult
   :exclude-members: __init__

Preconditioned Conjugate Gradient
---------------------------------

.. autoclass:: rlaopt.solvers.PCG
   :members:

.. autopydantic_model:: rlaopt.solvers.PCGConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.PCGStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.PCGResult
   :exclude-members: __init__

Convergence Status
------------------

.. autoclass:: rlaopt.solvers.ConvergenceStatus
   :members:
   :undoc-members:
