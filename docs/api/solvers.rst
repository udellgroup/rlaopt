Solvers Module
===============

.. automodule:: rlaopt.solvers

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

Preconditioned Conjugate Gradient
----------------------------------

.. autoclass:: rlaopt.solvers.PCG
   :members:
   :undoc-members:
   :show-inheritance:

.. autopydantic_model:: rlaopt.solvers.PCGConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.solvers.PCGStoppingCriteria
   :inherited-members: BaseModel



