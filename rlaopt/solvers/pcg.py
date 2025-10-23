"""Preconditioned Conjugate Gradient solver implementation."""

from typing import Callable

import torch

from rlaopt._typing import LinSysState
from rlaopt.linalg import (
    IdentityConfig,
    LinSys,
    Preconditioner,
    PreconditionerConfig,
    get_preconditioner,
)
from rlaopt.solvers.configs_base import LinSysSolverConfig
from rlaopt.solvers.solver_base import LinSysSolver


class PCGConfig(LinSysSolverConfig):
    """Configuration for the Preconditioned Conjugate Gradient solver.

    Attributes:
        tol: Tolerance for convergence (relative residual norm).
        preconditioner_config: Configuration for the preconditioner to use.
    """

    preconditioner_config: PreconditionerConfig = IdentityConfig()


class PCGState(LinSysState):
    """State container for the PCG solver.

    Attributes:
        r: Residual vector (B - AW).
        z: Preconditioned residual (P_inv @ r).
        p: Search direction.
        rz: Inner product r^T @ z (for non-converged components).
        res_norm: Current residual norm per component.
        mask: Boolean mask indicating which components have not yet converged.
        iter_: Current iteration count, starting from 0.
    """

    r: torch.Tensor
    z: torch.Tensor
    p: torch.Tensor
    rz: torch.Tensor
    res_norm: torch.Tensor
    mask: torch.Tensor
    iter_: int = 0


class PCG(LinSysSolver):
    """Preconditioned Conjugate Gradient solver for linear systems.

    Solves linear systems of the form:
        AW = B
    where A is a symmetric positive-definite matrix.

    The PCG method uses a preconditioner to improve convergence. The algorithm
    iteratively refines the solution by moving along conjugate search directions
    that are scaled by the preconditioner.
    """

    def __init__(self, config: PCGConfig, lin_sys: LinSys):
        """Initialize the PCG solver.

        Args:
            config (PCGConfig): Configuration for the solver.
            lin_sys (LinSys): The linear system to solve.
        """
        super().__init__(config, lin_sys)
        P = get_preconditioner(config.preconditioner_config, lin_sys.A, lin_sys.device)
        self._init_state = _build_init_state(lin_sys, P, config.tol)
        self._step = _build_step(lin_sys, P, config.tol)

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


def _compute_convergence_mask(
    res_norm: torch.Tensor, lin_sys: LinSys, tol: float
) -> torch.Tensor:
    epsilon = tol * lin_sys.rhs_norm
    mask = res_norm > epsilon
    return mask


def _build_init_state(
    lin_sys: LinSys, P: Preconditioner, tol: float
) -> Callable[[torch.Tensor], PCGState]:
    def init_state(params: torch.Tensor) -> PCGState:
        """Initialize the PCG solver state.

        Args:
            params: Initial parameters (solution estimate).

        Returns:
            Initial solver state.
        """
        # Compute initial residual: r = B - A @ params
        r = lin_sys.compute_residual(params)

        # Apply preconditioner
        z = P.inv @ r

        # Initialize search direction
        p = z.clone()

        # Compute initial residual norm per component
        res_norm = torch.linalg.norm(r, dim=0, ord=2)

        # Initialize mask
        mask = _compute_convergence_mask(res_norm, lin_sys, tol)

        # Compute r^T @ z as a matrix (rz[i,j] corresponds to components i and j)
        rz = r.T @ z

        return PCGState(r=r, z=z, p=p, rz=rz, res_norm=res_norm, mask=mask, iter_=0)

    return init_state


def _build_step(
    lin_sys: LinSys,
    P: Preconditioner,
    tol: float,
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
        params[:, mask] += p_masked @ alpha_masked

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

        # Update mask based on convergence
        mask_new = _compute_convergence_mask(res_norm_new, lin_sys, tol)

        new_state = PCGState(
            r=r_new,
            z=z_new,
            p=p_new,
            rz=rz_new,
            res_norm=res_norm_new,
            mask=mask_new,
            iter_=state.iter_ + 1,
        )

        return params, new_state

    return step
