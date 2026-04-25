"""Helper module for building preconditioner update function."""

from dataclasses import replace
from typing import Callable

import torch
from linops import LinearOperator

from rlaopt.data import DataLoader
from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import (
    IdentityConfig,
    PreconditionerConfig,
    get_preconditioner,
    randomized_powering,
)
from rlaopt.splitting.sapphire_split import SapphireSplit

from .gradient_solver_states import SapphireState

DataBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def build_preconditioner_update(
    precond_config: PreconditionerConfig,
    op_split: SapphireSplit,
    precond_loader: DataLoader,
    precond_update_freq: int,
    device: torch.device,
    dtype: torch.dtype,
    update_stepsize: bool = True,
) -> Callable[[TensorDict, SapphireState], SapphireState]:
    """Build preconditioner update function.

    Returns:
        Callable for updating the preconditioner
    """

    def update_precond_fn(beta_value: TensorDict, state: SapphireState):
        if state.iter_ % precond_update_freq == 0:
            X_batch, y_batch, _ = precond_loader.get_batch()
            Hop = op_split.get_subsamp_hessian_linop(
                beta_value, X_batch, y_batch, device
            )

            P = get_preconditioner(precond_config, Hop, dtype)

            if update_stepsize:
                if isinstance(precond_config, IdentityConfig):
                    Aop = Hop
                else:
                    X_batch, y_batch, _ = precond_loader.get_batch()
                    Hop = op_split.get_subsamp_hessian_linop(
                        beta_value, X_batch, y_batch, device
                    )

                    def Aop(v: torch.Tensor) -> torch.Tensor:
                        return P.inv @ (Hop @ v + P.current_damping * v)

                state = _update_stepsize(state, Aop, Hop.shape, device, precond_config)

            return replace(state, P=P)
        else:
            return state

    return update_precond_fn


def _update_stepsize(
    state: SapphireState,
    Aop: Callable[[torch.Tensor], torch.Tensor] | LinearOperator,
    shape: tuple[int, int],
    device: torch.device,
    precond_config: PreconditionerConfig,
) -> SapphireState:
    L = randomized_powering(Aop, shape, device=device)

    if isinstance(precond_config, IdentityConfig):
        # Clamp so step size (1/L) never exceeds 100;
        # minibatch estimates can be overconfident
        L = max(L, 1e-2)
    else:
        L = 2 * L  # conservative safety factor for preconditioned operator

    eta_new = 1 / L

    return replace(state, eta=eta_new)
