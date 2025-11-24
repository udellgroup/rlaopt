"""ADMM implementation.

Our ADMM implementation is based on the description in
"GeNIOS: an (almost) second-order operator-splitting solver for
large-scale convex optimization" by Diamandis et al., 2023.
This implements an inexact ADMM solver that can handle large-scale
problems by solving the ADMM linear system approximately using
preconditioned conjugate gradient (PCG).
"""

from pydantic import Field

from rlaopt.linalg import (
    NystromConfig,
    PreconditionerConfig,
)
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria


class ADMMConfig(SolverConfig):
    """Configuration for the ADMM solver.

    Attributes:
        rho: Augmented Lagrangian penalty.
        rho_update_factor: Factor to update rho in primal-dual balancing.
        rho_update_threshold: Threshold for updating rho in primal-dual balancing.
        alpha: Over-relaxation parameter.
        sigma: Regularization parameter for the inexact ADMM linear system.
        gamma: Exponent for the linear system solve tolerance.
        preconditioner_config: Configuration for the preconditioner.
        preconditioner_update_freq: Frequency (in iterations) for
            updating the preconditioner.
    """

    rho: float = Field(
        1.0,
        description="Augmented Lagrangian penalty.",
    )
    rho_update_factor: float = Field(
        2.0,
        description="Factor to update rho in primal-dual balancing.",
    )
    rho_update_threshold: float = Field(
        10.0,
        description="Threshold for updating rho in primal-dual balancing.",
    )
    alpha: float = Field(
        1.6,
        description="Over-relaxation parameter.",
    )
    sigma: float = Field(
        1e-6,
        description="Regularization parameter for the inexact ADMM linear system.",
    )
    gamma: float = Field(
        1.2,
        description="Exponent for the linear system solve tolerance.",
    )
    preconditioner_config: PreconditionerConfig = NystromConfig(rank_init=50)
    preconditioner_update_freq: int = Field(
        20,
        description="Frequency (in iterations) for updating the preconditioner.",
    )


class ADMMStoppingCriteria(StoppingCriteria):
    """Stopping criteria for the ADMM solver."""

    eps_abs: float = Field(
        1e-4, description="Absolute tolerance for primal and dual residuals."
    )
    eps_rel: float = Field(
        1e-4, description="Relative tolerance for primal and dual residuals."
    )
    eps_infeas: float = Field(
        1e-8, description="Tolerance for infeasibility detection."
    )
