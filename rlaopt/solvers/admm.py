"""Alternating direction method of multipliers (ADMM) implementation.

Our ADMM implementation is based on the description in
"GeNIOS: an (almost) second-order operator-splitting solver for
large-scale convex optimization" by Diamandis et al., 2023.
This implements an inexact ADMM solver that can handle large-scale
problems by solving the ADMM linear system approximately using
preconditioned conjugate gradient (PCG).
"""

from dataclasses import dataclass
from typing import Callable
from warnings import warn

import torch
from pydantic import Field

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import (
    LinSys,
    NystromConfig,
    Preconditioner,
    PreconditionerConfig,
    get_preconditioner,
)
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria
from rlaopt.solvers.pcg import _PCG, PCGStoppingCriteria
from rlaopt.solvers.solver_base import (
    ConvergenceStatus,
    OptimResult,
    OptimSolver,
    SolverState,
)
from rlaopt.splitting import ADMMSplit

PCG_TOL_EPS = 1e-10


class ADMMConfig(SolverConfig):
    """Configuration for the ADMM solver.

    Attributes:
        rho: Augmented Lagrangian penalty at initialization.
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
        description="Augmented Lagrangian penalty at initialization.",
        gt=0.0,
    )
    rho_update_factor: float = Field(
        2.0,
        description="Factor to update rho in primal-dual balancing.",
        gt=1.0,
    )
    rho_update_threshold: float = Field(
        10.0,
        description="Threshold for updating rho in primal-dual balancing.",
        gt=0.0,
    )
    rho_update_freq: int = Field(
        25,
        description="Frequency (in iterations) for updating rho.",
        ge=1,
    )
    alpha: float = Field(
        1.6,
        description="Over-relaxation parameter.",
        gt=0.0,
        lt=2.0,
    )
    sigma: float = Field(
        1e-6,
        description="Regularization parameter for the inexact ADMM linear system.",
        ge=0.0,
    )
    gamma: float = Field(
        1.2,
        description="Exponent for the linear system solve tolerance.",
        gt=1.0,
    )
    preconditioner_config: PreconditionerConfig = Field(
        NystromConfig(rank_init=50, base_damping=0.0),
        description="Configuration for the linear system preconditioner.",
    )
    preconditioner_update_freq: int = Field(
        20,
        description="Frequency (in iterations) for updating the preconditioner.",
        ge=1,
    )


class ADMMStoppingCriteria(StoppingCriteria):
    """Stopping criteria for the ADMM solver."""

    eps_abs: float = Field(
        1e-4, description="Absolute tolerance for primal and dual residuals.", gt=0.0
    )
    eps_rel: float = Field(
        1e-4, description="Relative tolerance for primal and dual residuals.", gt=0.0
    )


@dataclass(frozen=True)
class ADMMState(SolverState):
    """State container for the ADMM solver.

    Attributes:
        iter_: Current iteration count.
        aux_variables: Auxiliary variables (z) in ADMM.
        dual_variables: Dual variables (u) in ADMM.
        primal_residual_norm: Norm of the primal residual.
        dual_residual_norm: Norm of the dual residual.
        rho: Current augmented Lagrangian penalty.
    """

    aux_variables: TensorDict
    dual_variables: TensorDict
    primal_residual_norm: float
    dual_residual_norm: float
    rho: float
    _preconditioner: Preconditioner | None = None


@dataclass(frozen=True)
class ADMMResult(OptimResult):
    """Result container for the ADMM solver.

    Attributes:
        variable_values: Optimized variable values.
        convergence_status: Status indicating how the solver terminated.
        num_iters: Number of iterations performed.
        primal_residual_norm: Norm of the primal residual.
        dual_residual_norm: Norm of the dual residual.
    """

    primal_residual_norm: float
    dual_residual_norm: float


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

        self._init_state = _build_init_state(op_split, config.rho)
        self._step = _build_step(
            op_split,
            config.rho_update_factor,
            config.rho_update_threshold,
            config.rho_update_freq,
            config.alpha,
            config.sigma,
            config.gamma,
            config.preconditioner_config,
            config.preconditioner_update_freq,
        )
        self._solve = lambda eps_abs, eps_rel, max_iters: _build_solve(
            op_split, self._init_state, self._step, eps_abs, eps_rel, max_iters
        )

    def init_state(self, variable_values: TensorDict) -> ADMMState:
        """Initialize the solver state.

        Args:
            variable_values: Initial variable values.

        Returns:
            Initial solver state.
        """
        return self._init_state(variable_values)

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
        return self._step(variables_values, state)

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
        solve_fn = self._solve(
            eps_abs=stopping_criteria.eps_abs,
            eps_rel=stopping_criteria.eps_rel,
            max_iters=stopping_criteria.max_iters,
        )
        return solve_fn(variable_values)


def _build_init_state(
    op_split: ADMMSplit, rho: float
) -> Callable[[TensorDict], ADMMState]:
    """Build function to initialize ADMM solver state.

    Args:
        op_split: ADMMSplit instance for the optimization problem.
        rho: Initial augmented Lagrangian penalty.

    Returns:
        Function that initializes the ADMM solver state.
    """

    def init_state(variable_values: TensorDict) -> ADMMState:
        aux_variables = op_split.r_variable_values
        dual_variables = op_split.init_dual_variables()
        primal_residual_norm, dual_residual_norm = (
            _compute_primal_and_dual_residual_norms(
                op_split,
                variable_values,
                aux_variables,
                dual_variables,
                rho,
            )
        )
        return ADMMState(
            iter_=0,
            aux_variables=aux_variables,
            dual_variables=dual_variables,
            primal_residual_norm=primal_residual_norm,
            dual_residual_norm=dual_residual_norm,
            rho=rho,
            _preconditioner=None,
        )

    return init_state


def _build_step(
    op_split: ADMMSplit,
    rho_update_factor: float,
    rho_update_threshold: float,
    rho_update_freq: int,
    alpha: float,
    sigma: float,
    gamma: float,
    preconditioner_config: PreconditionerConfig,
    preconditioner_update_freq: int,
) -> Callable[[TensorDict, ADMMState], tuple[TensorDict, ADMMState]]:
    """Build function to perform a single ADMM optimization step."""

    def step(
        variable_values: TensorDict, state: ADMMState
    ) -> tuple[TensorDict, ADMMState]:
        """Perform a single ADMM optimization step."""
        # Unpack state
        iter_ = state.iter_
        aux_variables = state.aux_variables
        dual_variables = state.dual_variables
        aux_variables_flat = aux_variables.to_flat_tensor()
        dual_variables_flat = dual_variables.to_flat_tensor()
        primal_residual_norm = state.primal_residual_norm
        dual_residual_norm = state.dual_residual_norm
        rho = state.rho
        preconditioner = state._preconditioner

        # Solve x-subproblem
        new_variable_values, preconditioner = _solve_x_subproblem(
            op_split,
            variable_values,
            aux_variables_flat,
            dual_variables_flat,
            rho,
            sigma,
            gamma,
            iter_,
            primal_residual_norm,
            dual_residual_norm,
            preconditioner,
            preconditioner_config,
            preconditioner_update_freq,
        )

        # Apply over-relaxation
        new_variable_values_flat = new_variable_values.to_flat_tensor()
        variable_values_overrelaxed_flat = _compute_overrelaxation(
            op_split, alpha, new_variable_values_flat, aux_variables_flat
        )

        # Update auxiliary variables
        new_aux_variables = _update_aux_variables(
            op_split,
            variable_values_overrelaxed_flat,
            dual_variables_flat,
            aux_variables,
            rho,
        )

        # Update dual variables
        new_dual_variables = _update_dual_variables(
            op_split,
            dual_variables,
            variable_values_overrelaxed_flat,
            new_aux_variables,
        )

        # Recompute residual norms
        new_primal_residual_norm, new_dual_residual_norm = (
            _compute_primal_and_dual_residual_norms(
                op_split,
                new_variable_values,
                new_aux_variables,
                new_dual_variables,
                rho,
            )
        )

        # Update rho using primal-dual balancing
        rho, new_dual_variables = _update_rho(
            rho,
            new_dual_variables,
            new_primal_residual_norm,
            new_dual_residual_norm,
            iter_,
            rho_update_factor,
            rho_update_threshold,
            rho_update_freq,
        )

        # Return updated state
        new_state = ADMMState(
            iter_=iter_ + 1,
            aux_variables=new_aux_variables,
            dual_variables=new_dual_variables,
            primal_residual_norm=new_primal_residual_norm,
            dual_residual_norm=new_dual_residual_norm,
            rho=rho,
            _preconditioner=preconditioner,
        )

        return new_variable_values, new_state

    return step


def _build_solve(
    op_split: ADMMSplit,
    init_state_fn: Callable[[TensorDict], ADMMState],
    step_fn: Callable[[TensorDict, ADMMState], tuple[TensorDict, ADMMState]],
    eps_abs: float,
    eps_rel: float,
    max_iters: int,
) -> Callable[[TensorDict | None], ADMMResult]:
    """Build the solve function with ADMM stopping criteria.

    Implements Boyd et al. stopping criteria:
    - Primal feasibility: ||r_primal|| <= eps_primal
    - Dual feasibility: ||r_dual|| <= eps_dual
    where:
    - eps_primal = sqrt(m) * eps_abs + eps_rel * max(||Ax||, ||z||, ||b||)
    - eps_dual = sqrt(n) * eps_abs + eps_rel * ||rho * A^T * u||
    """

    def solve(var_vals: TensorDict | None = None) -> ADMMResult:
        """Solve the optimization problem using ADMM."""
        if var_vals is None:
            var_vals = op_split.f_and_affine_variable_values

        state = init_state_fn(var_vals)

        m = op_split.A.shape[0]  # Constraint dimension
        n = op_split.A.shape[1]  # Variable dimension

        # Main iteration loop
        while state.iter_ < max_iters:
            var_vals, state = step_fn(var_vals, state)

            # Compute stopping criteria thresholds (Boyd et al., Section 3.3.1)

            # Primal threshold
            Ax_norm = torch.linalg.norm(op_split.A @ var_vals.to_flat_tensor()).item()
            z_norm = state.aux_variables.flat_norm().item()
            b_norm = torch.linalg.norm(op_split.b).item()
            eps_primal = (m**0.5) * eps_abs + eps_rel * max(Ax_norm, z_norm, b_norm)

            # Dual threshold
            ATu_norm = torch.linalg.norm(
                state.rho * op_split.A_T @ state.dual_variables.to_flat_tensor()
            ).item()
            eps_dual = (n**0.5) * eps_abs + eps_rel * ATu_norm

            # Check convergence
            if (
                state.primal_residual_norm <= eps_primal
                and state.dual_residual_norm <= eps_dual
            ):
                convergence_status = ConvergenceStatus.CONVERGED
                break
        else:
            # Max iterations reached without convergence
            convergence_status = ConvergenceStatus.NOT_CONVERGED

        return ADMMResult(
            variable_values=var_vals,
            convergence_status=convergence_status,
            num_iters=state.iter_,
            primal_residual_norm=state.primal_residual_norm,
            dual_residual_norm=state.dual_residual_norm,
        )

    return solve


def _solve_x_subproblem(
    op_split: ADMMSplit,
    variable_values: TensorDict,
    aux_variables_flat: torch.Tensor,
    dual_variables_flat: torch.Tensor,
    rho: float,
    sigma: float,
    gamma: float,
    iter_: int,
    primal_residual_norm: float,
    dual_residual_norm: float,
    preconditioner: Preconditioner | None,
    preconditioner_config: PreconditionerConfig,
    preconditioner_update_freq: int,
) -> tuple[TensorDict, Preconditioner]:
    """Solve the x-subproblem approximately using PCG.

    Returns:
        Tuple of updated variable values and preconditioner.
    """
    variables_values_flat = variable_values.to_flat_tensor()
    rhs = -op_split.grad_f(variable_values).to_flat_tensor()
    rhs += op_split.hvp_f(variable_values, variables_values_flat)
    rhs += sigma * variables_values_flat
    rhs += rho * op_split.A_T @ (aux_variables_flat - dual_variables_flat + op_split.b)
    lin_sys = LinSys(
        op_split.hvp_f_ATA_linop(variable_values, rho),
        B=rhs,
        reg=sigma,
        w=variables_values_flat,
    )

    # Compute preconditioner if needed
    if iter_ % preconditioner_update_freq == 0 or preconditioner is None:
        preconditioner = get_preconditioner(
            preconditioner_config,
            lin_sys.A,
            lin_sys.dtype,
        )

    # Solve the linear system using PCG
    pcg_solver = _PCG(lin_sys, preconditioner=preconditioner)
    rel_tol = (
        min((primal_residual_norm * dual_residual_norm + PCG_TOL_EPS) ** 0.5, 1.0)
        / (iter_ + 1) ** gamma
    )
    stopping_criteria = PCGStoppingCriteria(
        max_iters=lin_sys.A.shape[0],
        tol=rel_tol,
    )
    pcg_solve_result = pcg_solver.solve(stopping_criteria=stopping_criteria)

    if pcg_solve_result.convergence_status != ConvergenceStatus.CONVERGED:
        warn(
            f"PCG for ADMM did not converge in iteration {iter_}. "
            f"Status: {pcg_solve_result.convergence_status}."
            "Consider changing the preconditioner."
        )

    # Need to squeeze last dimension since PCG returns 2D tensors
    new_variable_values_flat = pcg_solve_result.solution.squeeze(-1)
    new_variable_values = variable_values.from_flat_tensor(new_variable_values_flat)

    return new_variable_values, preconditioner


def _compute_overrelaxation(
    op_split: ADMMSplit,
    alpha: float,
    new_variable_values_flat: torch.Tensor,
    aux_variables_flat: torch.Tensor,
) -> torch.Tensor:
    """Apply over-relaxation to the variable values.

    Returns:
        Over-relaxed variable values.
    """
    return alpha * op_split.A @ new_variable_values_flat + (1 - alpha) * (
        aux_variables_flat + op_split.b
    )


def _update_aux_variables(
    op_split: ADMMSplit,
    variable_values_overrelaxed_flat: torch.Tensor,
    dual_variables_flat: torch.Tensor,
    aux_variables: TensorDict,
    rho: float,
) -> TensorDict:
    """Update auxiliary variables using proximal operators.

    Returns:
        Updated auxiliary variables.
    """
    aux_variables_intermediate_flat = (
        variable_values_overrelaxed_flat + dual_variables_flat - op_split.b
    )
    aux_variables_intermediate = aux_variables.from_flat_tensor(
        aux_variables_intermediate_flat
    )
    return op_split.prox(aux_variables_intermediate, 1.0 / rho)


def _update_dual_variables(
    op_split: ADMMSplit,
    dual_variables: TensorDict,
    variable_values_overrelaxed_flat: torch.Tensor,
    new_aux_variables: TensorDict,
) -> TensorDict:
    """Update dual variables.

    Returns:
        Updated dual variables.
    """
    dual_variables_flat = dual_variables.to_flat_tensor()
    new_dual_variables_flat = (
        dual_variables_flat
        + variable_values_overrelaxed_flat
        - new_aux_variables.to_flat_tensor()
        - op_split.b
    )
    return dual_variables.from_flat_tensor(new_dual_variables_flat)


def _update_rho(
    rho: float,
    dual_variables: TensorDict,
    primal_residual_norm: float,
    dual_residual_norm: float,
    iter_: int,
    rho_update_factor: float,
    rho_update_threshold: float,
    rho_update_freq: int,
) -> tuple[float, TensorDict]:
    """Update penalty parameter rho using primal-dual balancing.

    Returns:
        Tuple of updated rho and dual variables.
    """
    if (iter_ + 1) % rho_update_freq == 0:
        if primal_residual_norm > rho_update_threshold * dual_residual_norm:
            rho *= rho_update_factor
            dual_variables = dual_variables / rho_update_factor
        elif dual_residual_norm > rho_update_threshold * primal_residual_norm:
            rho /= rho_update_factor
            dual_variables = dual_variables * rho_update_factor

    return rho, dual_variables


def _compute_primal_and_dual_residual_norms(
    op_split: ADMMSplit,
    variable_values: TensorDict,
    aux_variables: TensorDict,
    dual_variables: TensorDict,
    rho: float,
) -> tuple[float, float]:
    """Compute the primal and dual residual norms for ADMM."""
    primal_residual = (
        op_split.A @ variable_values.to_flat_tensor()
        - aux_variables.to_flat_tensor()
        - op_split.b
    )
    dual_residual = (
        op_split.grad_f(variable_values).to_flat_tensor()
        + rho * op_split.A_T @ dual_variables.to_flat_tensor()
    )
    primal_residual_norm = torch.linalg.norm(primal_residual).item()
    dual_residual_norm = torch.linalg.norm(dual_residual).item()
    return primal_residual_norm, dual_residual_norm
