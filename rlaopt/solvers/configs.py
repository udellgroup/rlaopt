"""Configuration classes for solver algorithms."""

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, frozen=False)
class SolverConfig:
    """Base configuration class for solver algorithms."""

    pass


@dataclass(kw_only=True, frozen=False)
class ProxGradConfig(SolverConfig):
    """Configuration for the proximal gradient solver.

    Attributes:
        eta: Step size for the gradient update.
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence.
        use_acceleration: Whether to use acceleration techniques.
        use_linesearch: Whether to use line search for step size selection.
    """

    eta: torch.Tensor = None
    max_iters: int = 5000
    tol: float = 1e-4
    use_acceleration: bool = False
    use_linesearch: bool = True
