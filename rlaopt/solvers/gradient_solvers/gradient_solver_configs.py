"""Solver configs module for gradient based solvers."""

from typing import Literal

from pydantic import Field

from rlaopt.linalg.preconditioners.identity import IdentityConfig
from rlaopt.linalg.preconditioners.preconditioner import PreconditionerConfig
from rlaopt.solvers.configs_base import SolverConfig


class GradSolverConfig(SolverConfig):
    """Base configuration for gradient solvers."""

    eta0: float


class ProxGradConfig(GradSolverConfig):
    """Configuration for proximal gradient solvers."""

    eta0: float = Field(default=1.0, gt=0.0)
    use_acceleration: bool = False
    use_linesearch: bool = True


class SapphireConfig(GradSolverConfig):
    """Configuration for Sapphire solver."""

    # Base optimizer specification.
    base_method: Literal["saga", "svrg", "sgd"] = "svrg"

    # Stepsize selection
    eta0: float = Field(default=0.25, gt=0.0)
    auto_update_stepsize: bool = True

    # Preconditioner hyperparameters
    precond_config: PreconditionerConfig = IdentityConfig()
    precond_batch_size: int = Field(default=256, gt=0)
    precond_update_freq: int = Field(default=1, gt=0)
    subproblem_iters: int = Field(default=20, gt=0)

    snapshot_update_freq: int = Field(default=1, gt=0)
    check_termination_freq: int = Field(default=1, gt=0)
