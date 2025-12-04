Linear Algebra Module
=====================

.. automodule:: rlaopt.linalg

Linear System Solver
--------------------

.. autoclass:: rlaopt.linalg.LinSys
   :members:
   :undoc-members:
   :show-inheritance:

Preconditioner Configuration
-----------------------------

.. autopydantic_model:: rlaopt.linalg.PreconditionerConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.linalg.IdentityConfig
   :inherited-members: BaseModel

.. autopydantic_model:: rlaopt.linalg.NystromConfig
   :inherited-members: BaseModel

