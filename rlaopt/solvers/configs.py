"""Configuration classes for solver algorithms."""

from pydantic import BaseModel, ConfigDict, Field


class SolverConfig(BaseModel):
    """Base configuration class for solver algorithms."""

    model_config = ConfigDict(extra="forbid")


class LinSysSolverConfig(SolverConfig):
    """Base configuration class for linear system solvers.

    Attributes:
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence (relative residual norm).
    """

    max_iters: int = Field(default=1000, gt=0)
    tol: float = Field(default=1e-6, gt=0)
