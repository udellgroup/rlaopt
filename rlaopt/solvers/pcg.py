"""Preconditioned Conjugate Gradient solver implementation."""

import torch

from rlaopt._typing import LinSysState
from rlaopt.linalg import LinSys
from rlaopt.solvers.configs_base import LinSysSolverConfig
from rlaopt.solvers.solver_base import LinSysSolver


class PCGConfig(LinSysSolverConfig):
    """Configuration for the Preconditioned Conjugate Gradient solver.

    Attributes:
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence (relative residual norm).
        P: Preconditioner matrix. If None, uses identity (no preconditioning).
        P_inv: Inverse of the preconditioner matrix. If None, uses identity.
    """

    P: torch.Tensor | None = None
    P_inv: torch.Tensor | None = None


class PCGState(LinSysState):
    """State container for the PCG solver.

    Attributes:
        r: Residual vector (B - (A + reg*I)w).
        z: Preconditioned residual (P_inv @ r).
        p: Search direction.
        rz: Inner product r^T @ z.
        res_norm: Current residual norm.
        iter_: Current iteration count, starting from 0.
    """

    r: torch.Tensor
    z: torch.Tensor
    p: torch.Tensor
    rz: torch.Tensor
    res_norm: torch.Tensor
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

    def init_state(self, lin_sys: LinSys) -> PCGState:
        """Initialize the solver state.

        Args:
            lin_sys: The linear system to solve.

        Returns:
            Initial solver state.
        """
        return _init_pcg_state(lin_sys, self.config.P_inv)

    def step(self, lin_sys: LinSys, state: PCGState) -> PCGState:
        """Perform a single PCG iteration step.

        Args:
            lin_sys: The linear system to solve.
            state: Current solver state.

        Returns:
            Updated state after one iteration.
        """
        return _pcg_step(lin_sys, state, self.config.P_inv)

    def solve(self, lin_sys: LinSys) -> torch.Tensor:
        """Solve the linear system using PCG.

        This method performs iterative refinement until convergence criteria
        are met (either tolerance is achieved or max iterations reached).

        Args:
            lin_sys: The linear system to solve.

        Returns:
            Solution tensor w satisfying (A + reg*I)w = B.
        """
        return _pcg(lin_sys, self.config)


def _pcg(lin_sys: LinSys, config: PCGConfig) -> torch.Tensor:
    """Solve a linear system using the Preconditioned Conjugate Gradient method.

    The PCG method iteratively solves systems of the form (A + reg*I)w = B
    where A is positive-definite. It uses a preconditioner to accelerate
    convergence by transforming the system to have better conditioning.

    Args:
        lin_sys: The linear system to solve containing A, B, reg, and initial w.
        config: Configuration parameters including max iterations, tolerance,
            and preconditioner.

    Returns:
        Solution tensor w satisfying (A + reg*I)w = B (within tolerance).
    """
    # Unpack config
    max_iters, tol, P_inv = config.max_iters, config.tol, config.P_inv

    # Initialize state
    state = _init_pcg_state(lin_sys, P_inv)

    # Get convergence tolerance
    epsilon = tol * torch.linalg.norm(lin_sys.B, dim=0, ord=2)

    # Solver loop
    while (state.res_norm > epsilon).any() and state.iter_ < max_iters:
        state = _pcg_step(lin_sys, state, P_inv)

    return lin_sys.w.data


def _init_pcg_state(lin_sys: LinSys, P_inv: torch.Tensor | None = None) -> PCGState:
    """Initialize the PCG solver state.

    Args:
        lin_sys: The linear system to solve.
        P_inv: Inverse of the preconditioner matrix. If None, uses identity.

    Returns:
        Initial PCG state with residuals and search directions.
    """
    # Compute initial residual: r = B - (A + reg*I)w
    r = lin_sys.B - lin_sys.forward(lin_sys.w)

    # Apply preconditioner: z = P_inv @ r
    if P_inv is not None:
        z = P_inv @ r
    else:
        z = r.clone()

    # Initialize search direction
    p = z.clone()

    # Compute r^T @ z
    rz = (r * z).sum(dim=0, keepdim=True)
    if rz.ndim == 2:
        rz = rz.squeeze(0)

    # Compute initial residual norm
    res_norm = torch.linalg.norm(r, dim=0, ord=2)

    return PCGState(r=r, z=z, p=p, rz=rz, res_norm=res_norm, iter_=0)


def _pcg_step(
    lin_sys: LinSys, state: PCGState, P_inv: torch.Tensor | None = None
) -> PCGState:
    """Perform a single PCG iteration.

    Args:
        lin_sys: The linear system to solve.
        state: Current solver state.
        P_inv: Inverse of the preconditioner matrix. If None, uses identity.

    Returns:
        Updated state after one PCG iteration.
    """
    # Compute A @ p (apply linear operator to search direction)
    Ap = lin_sys.forward(state.p)

    # Compute step size: alpha = (r^T @ z) / (p^T @ A @ p)
    pAp = (state.p * Ap).sum(dim=0, keepdim=True)
    if pAp.ndim == 2:
        pAp = pAp.squeeze(0)

    alpha = state.rz / pAp

    # Expand alpha for broadcasting if needed
    if alpha.ndim == 1 and lin_sys.w.ndim == 2:
        alpha = alpha.unsqueeze(0)

    # Update solution: w = w + alpha * p (update in place)
    lin_sys.w.data.add_(alpha * state.p)

    # Update residual: r = r - alpha * A @ p
    r_new = state.r - alpha * Ap

    # Apply preconditioner to new residual: z = P_inv @ r
    if P_inv is not None:
        z_new = P_inv @ r_new
    else:
        z_new = r_new.clone()

    # Compute new r^T @ z
    rz_new = (r_new * z_new).sum(dim=0, keepdim=True)
    if rz_new.ndim == 2:
        rz_new = rz_new.squeeze(0)

    # Compute beta = (r_new^T @ z_new) / (r^T @ z)
    beta = rz_new / state.rz

    # Expand beta for broadcasting if needed
    if beta.ndim == 1 and state.p.ndim == 2:
        beta = beta.unsqueeze(0)

    # Update search direction: p = z + beta * p
    p_new = z_new + beta * state.p

    # Compute new residual norm
    res_norm_new = torch.linalg.norm(r_new, dim=0, ord=2)

    return PCGState(
        r=r_new,
        z=z_new,
        p=p_new,
        rz=rz_new,
        res_norm=res_norm_new,
        iter_=state.iter_ + 1,
    )
