"""Configuration classes for solver algorithms."""

from pydantic import BaseModel, ConfigDict, Field


class SolverConfig(BaseModel):
    """Base configuration class for solver algorithms."""

    model_config = ConfigDict(extra="forbid")


class ProxGradConfig(SolverConfig):
    """Configuration for the proximal gradient solver.

    Attributes:
        eta: Step size for the gradient update.
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence.
        use_acceleration: Whether to use acceleration techniques.
        use_linesearch: Whether to use line search for step size selection.
    """

    eta: float = Field(default=1.0, gt=0)
    max_iters: int = Field(default=5000, gt=0)
    tol: float = Field(default=1e-4, gt=0)
    use_acceleration: bool = False
    use_linesearch: bool = True
