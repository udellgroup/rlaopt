"""Preconditioned Conjugate Gradient solver implementation."""

from dataclasses import dataclass, replace
from typing import Callable

import torch
from pydantic import Field

from rlaopt.linalg import (
    IdentityConfig,
    LinSys,
    Preconditioner,
    PreconditionerConfig,
    get_preconditioner,
)
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria
from rlaopt.solvers.solver_base import LinSysSolver, SolverState


class PCGConfig(SolverConfig):
    """Configuration for the Preconditioned Conjugate Gradient solver.

    Attributes:
        preconditioner_config: Configuration for the preconditioner to use.
    """

    preconditioner_config: PreconditionerConfig = IdentityConfig()


class PCGStoppingCriteria(StoppingCriteria):
    """Stopping criteria specific to the PCG solver.

    Attributes:
        max_iters: Maximum number of iterations.
        tol: Relative tolerance for convergence.
    """

    tol: float = Field(default=1e-6, gt=0)


@dataclass(frozen=True)
class PCGState(SolverState):
    """State container for the PCG solver.

    Attributes:
        iter_: Current iteration count.
        r: Residual vector (B - AW).
        z: Preconditioned residual (P_inv @ r).
        p: Search direction.
        rz: Inner product r^T @ z (for non-converged components).
        res_norm: Current residual norm per component.
        mask: Boolean mask indicating which components have not yet converged.
            If there is no tolerance set, all components are considered active.
        tol: Optional tolerance for convergence checking. When None, all components
            are considered active (mask is not updated based on convergence).
    """

    r: torch.Tensor
    z: torch.Tensor
    p: torch.Tensor
    rz: torch.Tensor
    res_norm: torch.Tensor
    mask: torch.Tensor
    tol: float | None = None


class PCG(LinSysSolver):
    """Block Preconditioned Conjugate Gradient solver for linear systems.

    Solves linear systems of the form:
        AW = B
    where A is a symmetric positive-definite matrix.

    The PCG method uses a preconditioner to improve convergence. The algorithm
    iteratively refines the solution by moving along conjugate search directions
    that are scaled by the preconditioner.
    """

    def __init__(self, lin_sys: LinSys, config: PCGConfig):
        """Initialize the PCG solver.

        Args:
            lin_sys (LinSys): The linear system to solve.
            config (PCGConfig): Configuration for the solver.
        """
        super().__init__(lin_sys, config)
        P = get_preconditioner(config.preconditioner_config, lin_sys.A, lin_sys.dtype)
        self._init_state = _build_init_state(lin_sys, P)
        self._step = _build_step(lin_sys, P)
        self._solve = lambda tol, max_iters: _build_solve(
            lin_sys, self._init_state, self._step, tol, max_iters
        )

    def init_state(self, params: torch.Tensor) -> PCGState:
        """Initialize the solver state.

        Args:
            params: Initial parameters (solution estimate).

        Returns:
            Initial solver state.
        """
        return self._init_state(params)

    def step(
        self, params: torch.Tensor, state: PCGState
    ) -> tuple[torch.Tensor, PCGState]:
        """Perform a single PCG iteration step.

        Args:
            params: Current parameters (solution estimate).
            state: Current solver state.

        Returns:
            Tuple of updated parameters and solver state.
        """
        return self._step(params, state)

    def solve(
        self,
        params: torch.Tensor | None = None,
        stopping_criteria: PCGStoppingCriteria = PCGStoppingCriteria(),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the linear system AW = B using PCG.

        Args:
            params: Initial parameters (solution estimate). If None, defaults to
                parameters in linear system.
            stopping_criteria: Criteria to determine when to stop the solver.
                Defaults to PCGStoppingCriteria().

        Returns:
            Tuple of solution parameters and final residual norm.
        """
        solve_fn = self._solve(
            tol=stopping_criteria.tol, max_iters=stopping_criteria.max_iters
        )
        return solve_fn(params)


def _compute_convergence_mask(
    res_norm: torch.Tensor, lin_sys: LinSys, tol: float
) -> torch.Tensor:
    epsilon = tol * lin_sys.rhs_norm
    mask = res_norm > epsilon
    return mask


def _build_init_state(
    lin_sys: LinSys, P: Preconditioner
) -> Callable[[torch.Tensor], PCGState]:
    def init_state(params: torch.Tensor) -> PCGState:
        """Initialize the PCG solver state.

        Args:
            params: Initial parameters (solution estimate).

        Returns:
            Initial solver state with all components marked as active (mask=True).
        """
        # Compute initial residual: r = B - A @ params
        r = lin_sys.compute_residual(params)

        # Apply preconditioner
        z = P.inv @ r

        # Initialize search direction
        p = z.clone()

        # Compute initial residual norm per component
        res_norm = torch.linalg.norm(r, dim=0, ord=2)

        # Initialize mask to all True (all components are active)
        # When tolerance is set via solve(), the mask will be updated in step()
        mask = torch.ones(r.shape[1], dtype=torch.bool, device=r.device)

        # Compute r^T @ z as a matrix (rz[i,j] corresponds to components i and j)
        rz = r.T @ z

        return PCGState(iter_=0, r=r, z=z, p=p, rz=rz, res_norm=res_norm, mask=mask)

    return init_state


def _build_step(
    lin_sys: LinSys,
    P: Preconditioner,
) -> Callable[[torch.Tensor, PCGState], tuple[torch.Tensor, PCGState]]:
    def step(
        params: torch.Tensor,
        state: PCGState,
    ) -> tuple[torch.Tensor, PCGState]:
        """Perform a single PCG iteration step.

        Args:
            params: Current parameters (solution estimate).
            state: Current solver state.

        Returns:
            Tuple of updated parameters and solver state.
        """
        # Get current mask
        mask = state.mask

        # If all components have converged, return unchanged state
        if not mask.any():
            return params, state

        # Apply mask to work only with non-converged components
        p_masked = state.p[:, mask]
        rz_masked = state.rz[mask][:, mask]

        # Compute A @ p only for non-converged directions
        Ap_masked = lin_sys(p_masked)

        # Compute alpha for active components
        alpha_masked = torch.linalg.solve(p_masked.T @ Ap_masked, rz_masked)

        # Only update the active parts of the solution
        params_new = params.clone()
        params_new[:, mask] += p_masked @ alpha_masked

        # Update residual for active components
        r_new = state.r.clone()
        r_new[:, mask] -= Ap_masked @ alpha_masked

        # Apply preconditioner to new residual for active components
        z_new_masked = P.inv @ r_new[:, mask]

        # Update z with new values for active components
        z_new = state.z.clone()
        z_new[:, mask] = z_new_masked

        # Compute new rz for active components
        rz_new_masked = r_new[:, mask].T @ z_new_masked

        # Compute beta for active components
        beta_masked = torch.linalg.solve(rz_masked, rz_new_masked)

        # Update search direction for active components
        p_new = state.p.clone()
        p_new[:, mask] = z_new_masked + p_masked @ beta_masked

        # Update rz matrix
        rz_new = torch.zeros_like(state.rz)
        rz_new[torch.outer(mask, mask)] = rz_new_masked.flatten()

        # Compute new residual norm
        res_norm_new = torch.linalg.norm(r_new, dim=0, ord=2)

        # Update mask based on convergence if tolerance is set
        if state.tol is not None:
            mask_new = _compute_convergence_mask(res_norm_new, lin_sys, state.tol)
        else:
            # If no tolerance is set, keep all components active
            mask_new = mask

        new_state = PCGState(
            iter_=state.iter_ + 1,
            r=r_new,
            z=z_new,
            p=p_new,
            rz=rz_new,
            res_norm=res_norm_new,
            mask=mask_new,
            tol=state.tol,
        )

        return params_new, new_state

    return step


def _build_solve(
    lin_sys: LinSys,
    init_state_fn: Callable[[torch.Tensor], PCGState],
    step_fn: Callable[[torch.Tensor, PCGState], tuple[torch.Tensor, PCGState]],
    tol: float,
    max_iters: int,
) -> Callable[[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
    """Build the solve function with stopping criteria."""

    def solve(params: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the linear system AW = B.

        Args:
            params: Initial parameters. If None, defaults to zeros.

        Returns:
            Tuple of solution parameters and final residual norm.
        """
        if params is None:
            params = lin_sys.w.clone()

        # Initialize state without tolerance
        state = init_state_fn(params)

        # Set tolerance in state for convergence checking
        state = replace(state, tol=tol)

        # Iterate until convergence or max iterations
        while state.mask.any() and state.iter_ < max_iters:
            params, state = step_fn(params, state)

        # Return final params and residual norm
        return params, state.res_norm

    return solve
