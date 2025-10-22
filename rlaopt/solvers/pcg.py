"""Preconditioned Conjugate Gradient solver implementation."""

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
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence (relative residual norm).
        preconditioner_config: Configuration for the preconditioner to use.
    """

    preconditioner_config: PreconditionerConfig = IdentityConfig()


class PCGState(LinSysState):
    """State container for the PCG solver.

    Attributes:
        w: Current solution estimate.
        r: Residual vector (B - (A + reg*I)w).
        z: Preconditioned residual (P_inv @ r).
        p: Search direction.
        rz: Inner product r^T @ z (for non-converged components).
        res_norm: Current residual norm per component.
        mask: Boolean mask indicating which components have not yet converged.
        iter_: Current iteration count, starting from 0.
    """

    w: torch.Tensor
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
        (A + reg*I)w = B
    where A is a positive-definite matrix and reg is a regularization parameter.

    The PCG method uses a preconditioner to improve convergence. The algorithm
    iteratively refines the solution by moving along conjugate search directions
    that are scaled by the preconditioner.
    """

    def __init__(self, config: PCGConfig):
        """Initialize the PCG solver.

        Args:
            config: Configuration for the solver including preconditioner.
        """
        super().__init__(config)
        self.P = None

    def init_state(self, lin_sys: LinSys) -> PCGState:
        """Initialize the solver state.

        Args:
            lin_sys: The linear system to solve.

        Returns:
            Initial solver state.
        """
        self.P = get_preconditioner(
            self.config.preconditioner_config, lin_sys.A, lin_sys.device
        )
        return _init_pcg_state(lin_sys, self.P, self.config.tol)

    def step(self, lin_sys: LinSys, state: PCGState) -> PCGState:
        """Perform a single PCG iteration step.

        Args:
            lin_sys: The linear system to solve.
            state: Current solver state.

        Returns:
            Updated state after one iteration.
        """
        return _pcg_step(lin_sys, state, self.P, self.config.tol)


def _init_pcg_state(lin_sys: LinSys, P: Preconditioner, tol: float = 1e-6) -> PCGState:
    """Initialize the PCG solver state.

    Args:
        lin_sys: The linear system to solve.
        P: Preconditioner to use.
        tol: Tolerance for convergence.

    Returns:
        Initial PCG state with residuals and search directions.
    """
    # Initial solution estimate
    w = lin_sys.w.clone()

    # Compute initial residual: r = B - (A + reg*I)w
    r = lin_sys.B - lin_sys(w)

    # Apply preconditioner
    z = P.inv @ r

    # Initialize search direction
    p = z.clone()

    # Compute initial residual norm per component
    res_norm = torch.linalg.norm(r, dim=0, ord=2)

    # Initialize mask - all components start as not converged
    epsilon = tol * torch.linalg.norm(lin_sys.B, dim=0, ord=2)
    mask = res_norm > epsilon

    # Compute r^T @ z as a matrix (rz[i,j] corresponds to components i and j)
    rz = r.T @ z

    return PCGState(w=w, r=r, z=z, p=p, rz=rz, res_norm=res_norm, mask=mask, iter_=0)


def _pcg_step(
    lin_sys: LinSys,
    state: PCGState,
    P: Preconditioner,
    tol: float = 1e-6,
) -> PCGState:
    """Perform a single PCG iteration.

    Args:
        lin_sys: The linear system to solve.
        state: Current solver state.
        P: Preconditioner to use.
        tol: Tolerance for convergence.

    Returns:
        Updated state after one PCG iteration.
    """
    # Get current mask
    mask = state.mask

    # If all components have converged, return unchanged state
    if not mask.any():
        return state

    # Apply mask to work only with non-converged components
    p_masked = state.p[:, mask]
    rz_masked = state.rz[mask][:, mask]

    # Compute A @ p only for non-converged directions
    Ap_masked = lin_sys(p_masked)

    # Compute alpha for active components
    pAp_masked = (p_masked * Ap_masked).sum(dim=0)
    alpha_masked = torch.linalg.solve(
        torch.diag(pAp_masked), torch.diag(rz_masked)
    ).diag()

    # Expand alpha for broadcasting
    if alpha_masked.ndim == 1 and state.w.ndim == 2:
        alpha_masked = alpha_masked.unsqueeze(0)

    # Only update the active parts of the solution
    w_new = state.w.clone()
    w_new[:, mask] += p_masked * alpha_masked

    # Update residual for active components
    r_new = state.r.clone()
    r_new[:, mask] -= Ap_masked * alpha_masked

    # Apply preconditioner to new residual for active components
    z_new_masked = P.inv @ r_new[:, mask]

    # Update z with new values for active components
    z_new = state.z.clone()
    z_new[:, mask] = z_new_masked

    # Compute new rz for active components
    rz_new_masked = r_new[:, mask].T @ z_new_masked

    # Compute beta for active components
    beta_masked = torch.linalg.solve(rz_masked, rz_new_masked).diag()

    # Expand beta for broadcasting
    if beta_masked.ndim == 1 and state.p.ndim == 2:
        beta_masked = beta_masked.unsqueeze(0)

    # Update search direction for active components
    p_new = state.p.clone()
    p_new[:, mask] = z_new_masked + p_masked * beta_masked

    # Update rz matrix
    rz_new = torch.zeros_like(state.rz)
    rz_new[torch.outer(mask, mask)] = rz_new_masked.flatten()

    # Compute new residual norm
    res_norm_new = torch.linalg.norm(r_new, dim=0, ord=2)

    # Update mask based on convergence
    epsilon = tol * torch.linalg.norm(lin_sys.B, dim=0, ord=2)
    mask_new = res_norm_new > epsilon

    return PCGState(
        w=w_new,
        r=r_new,
        z=z_new,
        p=p_new,
        rz=rz_new,
        res_norm=res_norm_new,
        mask=mask_new,
        iter_=state.iter_ + 1,
    )
