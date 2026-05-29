"""Helper module for building preconditioner update function."""

from dataclasses import replace
from typing import Callable

import torch
from linops import LinearOperator

from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import (
    IdentityConfig,
    PreconditionerConfig,
    get_preconditioner,
    randomized_powering,
)

from .gradient_solver_states import GradSolverState

DataBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def build_preconditioner_update(
    precond_config: PreconditionerConfig,
    hessian_linop_fn: Callable[[TensorDict], LinearOperator],
    precond_update_freq: int,
    device: torch.device,
    dtype: torch.dtype,
    update_stepsize: bool = True,
) -> Callable[[TensorDict, GradSolverState], GradSolverState]:
    """Build preconditioner update function.

    Args:
        precond_config: Preconditioner configuration.
        hessian_linop_fn: Callable producing a Hessian linear operator at
            the current iterate. Caller decides how it is constructed (full,
            subsampled, etc.). Called once to build the preconditioner and
            again (when ``update_stepsize=True`` and ``precond_config`` is
            non-identity) for an independent step-size estimate.
        precond_update_freq: Iterations between preconditioner refreshes.
        device: Device on which the operator lives.
        dtype: Dtype used for preconditioner construction.
        update_stepsize: Whether to recompute ``state.eta`` from the
            preconditioned Hessian's spectral radius on each refresh.

    Returns:
        Callable for updating the preconditioner.
    """

    def update_precond_fn(beta_value: TensorDict, state: GradSolverState):
        if state.iter_ % precond_update_freq == 0:
            Hop = hessian_linop_fn(beta_value)

            P = get_preconditioner(precond_config, Hop, dtype)

            if update_stepsize:
                if isinstance(precond_config, IdentityConfig):
                    Aop = Hop
                else:
                    Hop = hessian_linop_fn(beta_value)

                    def Aop(v: torch.Tensor) -> torch.Tensor:
                        return P.inv @ (Hop @ v + P.current_damping * v)

                state = _update_stepsize(state, Aop, Hop.shape, device, precond_config)

            return replace(state, P=P)
        else:
            return state

    return update_precond_fn


def _update_stepsize(
    state: GradSolverState,
    Aop: Callable[[torch.Tensor], torch.Tensor] | LinearOperator,
    shape: tuple[int, int],
    device: torch.device,
    precond_config: PreconditionerConfig,
) -> GradSolverState:
    L = randomized_powering(Aop, shape, device=device)

    if isinstance(precond_config, IdentityConfig):
        # Clamp so step size (1/L) never exceeds 100;
        # minibatch estimates can be overconfident
        L = max(L, 1e-2)
    else:
        L = 2 * L  # conservative safety factor for preconditioned operator

    eta_new = 1 / L

    return replace(state, eta=eta_new)
