"""Alternating direction method of multipliers (ADMM) implementation.

Our ADMM implementation is based on the description in
"GeNIOS: an (almost) second-order operator-splitting solver for
large-scale convex optimization" by Diamandis et al., 2023.
This implements an inexact ADMM solver that can handle large-scale
problems by solving the ADMM linear system approximately using
preconditioned conjugate gradient (PCG).
"""

from dataclasses import dataclass

import torch
from pydantic import Field

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import (
    NystromConfig,
    PreconditionerConfig,
)
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria
from rlaopt.solvers.solver_base import OptimSolver, SolverResult, SolverState
from rlaopt.splitting import ADMMSplit


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
    preconditioner_config: PreconditionerConfig = Field(
        NystromConfig(rank_init=50),
        description="Configuration for the linear system preconditioner.",
    )
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


@dataclass(frozen=True)
class ADMMState(SolverState):
    """State container for the ADMM solver.

    Attributes:
        aux_variables: Auxiliary variables (z) in ADMM.
        dual_variables: Dual variables (u) in ADMM.
        primal_residual_norm: Norm of the primal residual.
        dual_residual_norm: Norm of the dual residual.
        rho: Current augmented Lagrangian penalty.
    """

    aux_variables: TensorDict
    dual_variables: TensorDict
    primal_residual_norm: torch.Tensor
    dual_residual_norm: torch.Tensor
    rho: float


@dataclass(frozen=True)
class ADMMResult(SolverResult):
    """Result container for the ADMM solver.

    Attributes:
        variable_values: Optimized variable values.
        primal_residual_norm: Norm of the primal residual.
        dual_residual_norm: Norm of the dual residual.
    """

    primal_residual_norm: torch.Tensor
    dual_residual_norm: torch.Tensor


class ADMM(OptimSolver):
    """Alternating Direction Method of Multipliers (ADMM) solver.

    Solves problems of the form:
        minimize f(x) + sum_i g_i(A_i x - b_i)
    where f is smooth (differentiable) and each g_i is proxable.
    """

    def __init__(self, obj: Expression, config: ADMMConfig):
        """Initialize the ADMM solver.

        Args:
            obj: The optimization problem to solve.
            config: Configuration for the ADMM solver.
        """
        if not isinstance(obj, Expression):
            raise ValueError("ADMM solver requires an Expression objective.")
        if not isinstance(config, ADMMConfig):
            raise ValueError("ADMM solver requires an ADMMConfig configuration.")
        super().__init__(obj, config)

        op_split = ADMMSplit(obj)

    def init_state(self, variable_values: TensorDict) -> ADMMState:
        """Initialize the solver state.

        Args:
            variable_values: Initial variable values.

        Returns:
            Initial solver state.
        """
        pass

    def step(
        self, variables_values: TensorDict, state: ADMMState
    ) -> tuple[TensorDict, ADMMState]:
        """Perform a single ADMM optimization step.

        Args:
            variables_values: Current variable values.
            state: Current ADMM solver state.

        Returns:
            Tuple of updated variable values and solver state.
        """
        pass

    def solve(
        self,
        variable_values: TensorDict | None = None,
        stopping_criteria: ADMMStoppingCriteria = ADMMStoppingCriteria(),
    ) -> ADMMResult:
        """Solve the optimization problem using ADMM.

        Args:
            variable_values: Initial variable values.
            stopping_criteria: Stopping criteria for the solver.

        Returns:
            ADMMResult: Result of the optimization containing optimized variable values
                among other metrics.
        """
        pass
