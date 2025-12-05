Solvers Module
===============

.. automodule:: rlaopt.solvers

ADMM
----

.. autoclass:: rlaopt.solvers.ADMM
   :members:
   :undoc-members:
   :show-inheritance:

.. autopydantic_model:: rlaopt.solvers.ADMMConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.ADMMStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.ADMMResult
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__

Proximal Gradient
-----------------

.. autoclass:: rlaopt.solvers.ProxGrad
   :members:
   :undoc-members:
   :show-inheritance:

.. autopydantic_model:: rlaopt.solvers.ProxGradConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.ProxGradStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.ProxGradResult
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__

Preconditioned Conjugate Gradient
---------------------------------

.. autoclass:: rlaopt.solvers.PCG
   :members:
   :undoc-members:
   :show-inheritance:

.. autopydantic_model:: rlaopt.solvers.PCGConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.PCGStoppingCriteria
   :inherited-members: BaseModel

.. autoclass:: rlaopt.solvers.PCGResult
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__
