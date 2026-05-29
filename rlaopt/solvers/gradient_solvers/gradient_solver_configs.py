"""Solver configs module for gradient based solvers."""

from typing import Literal

from pydantic import Field

from rlaopt.linalg import IdentityConfig, NystromConfig
from rlaopt.linalg.preconditioners.preconditioner import PreconditionerConfig
from rlaopt.solvers.configs_base import SolverConfig


class GradSolverConfig(SolverConfig):
    """Base configuration for gradient solvers."""

    eta: float

    precond_config: PreconditionerConfig = Field(
        default=IdentityConfig(),
        description="Preconditioner configuration. Defaults to identity"
        " (i.e. no preconditioning).",
    )

    subproblem_iters: int = Field(
        default=20,
        gt=0,
        description="Number of accelerated proximal gradient iterations "
        "performed to evaluate the scaled proximal operator. "
        "Not used if the problem is fully smooth or the preconditioner is identity.",
    )

    auto_update_stepsize: bool = Field(
        default=False,
        description="Boolean flag specifying whether to automatically "
        "update the stepsize based on an estimate of the local "
        "smoothness constant. Defaults to False.",
    )


class ProxGradConfig(GradSolverConfig):
    """Configuration for proximal gradient solvers."""

    eta: float = Field(
        default=1.0, gt=0.0, description="Step size for the gradient update."
    )

    use_acceleration: bool = Field(
        default=False, description="Whether to use Nesterov acceleration."
    )

    use_linesearch: bool = Field(
        default=True, description="Whether to use line search for step size selection."
    )


class SapphireConfig(GradSolverConfig):
    """Configuration for Sapphire solver."""

    # Base optimizer specification.
    base_method: Literal["saga", "svrg", "sgd"] = Field(
        default="saga", description="Base gradient method to use in Sapphire."
    )

    # Stepsize selection
    eta: float = Field(
        default=0.1,
        gt=0.0,
        description="Step size for the gradient update. "
        "Only used when auto_update_stepsize = False",
    )

    auto_update_stepsize: bool = Field(
        default=True,
        description="Boolean flag specifing whether to automatically "
        "update the stepsize based on an estimate of the local "
        "smoothness constant. Defaults to True.",
    )

    # Preconditioner hyperparameters
    precond_config: PreconditionerConfig = Field(
        default=NystromConfig(
            rank_init=10,
            error_tolerance=1e-1,
            base_damping=1e-3,
            damping_mode="adaptive",
        ),
        description="Config specifying preconditioner to be used in Sapphire."
        "Defaults to Nyström preconditioner.",
    )

    precond_update_freq: int = Field(
        default=2,
        gt=0,
        description="How frequently in epochs the preconditioner and "
        "stepsize (if auto_update_stepize = True) are updated. "
        "Defaults to 2 epochs.",
    )

    snapshot_update_freq: int = Field(
        default=1,
        gt=0,
        description="How frequently in epochs the SVRG snapshot is updated."
        "Default is one epoch. This is only relevant when base_method = 'svrg'.",
    )

    check_termination_freq: int = Field(
        default=1,
        gt=0,
        description="How frequently to compute criteria for checking termination. "
        "Default is one epoch.",
    )
