"""Solver states module for gradient based solvers."""

from dataclasses import dataclass

import torch

from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg.preconditioners.identity import Identity, IdentityConfig
from rlaopt.linalg.preconditioners.nystrom import Nystrom
from rlaopt.solvers.solver_base import SolverState
from rlaopt.splitting.prox_grad_split import ProxGradSplit
from rlaopt.splitting.sapphire_split import SapphireSplit

from .gradient_solver_configs import GradSolverConfig, ProxGradConfig, SapphireConfig


@dataclass(frozen=True)
class GradSolverState(SolverState):
    """Base state class for gradient-based solvers.

    Attributes:
        eta: Step size for gradient updates. Defaults to 1.0.
        err: Current error metric value. Defaults to infinity.
    """

    eta: float = 1.0
    err: float = torch.inf


@dataclass(frozen=True)
class ProxGradState(GradSolverState):
    """State class for proximal gradient methods.

    Extends GradSolverState with storage for previous iterate values,
    enabling momentum/acceleration schemes.

    Attributes:
        variable_values_prev: Previous iteration's variable values for momentum.
            None until first iteration completes.
        err: Current error metric value. Defaults to infinity.
    """

    variable_values_prev: TensorDict | None = None
    err: float = torch.inf


@dataclass(frozen=True)
class SGDState(GradSolverState):
    """State class for stochastic gradient descent methods.

    Extends GradSolverState with preconditioner support for variance-reduced
    and accelerated stochastic methods.

    Attributes:
        P: Preconditioner matrix (Nystrom approximation or Identity).
            Defaults to Identity.
        precond_update_freq: Frequency (in iterations) for updating the
            preconditioner. Defaults to 1.
    """

    P: Nystrom | Identity = Identity(IdentityConfig())
    precond_update_freq: int = 1


@dataclass(frozen=True)
class SAGAState(SGDState):
    """State class for SAGA (Stochastic Average Gradient Augmented) method.

    Extends SGDState with gradient table storage for variance reduction.
    SAGA maintains per-sample gradient information to reduce variance.

    Attributes:
        table: Tensor storing individual gradient components for each sample.
            Shape (n, ...) where n is the number of samples. None until initialized.
        grad_avg: Average of all gradients in the table. None until initialized.
    """

    table: torch.Tensor | None = None
    grad_avg: TensorDict | None = None


@dataclass(frozen=True)
class SVRGState(SGDState):
    """State class for SVRG (Stochastic Variance Reduced Gradient) method.

    Extends SGDState with snapshot storage for variance reduction. SVRG
    periodically computes full gradients at snapshot points and uses them
    as control variates.

    Attributes:
        snapshot: Variable values at the snapshot point. None until first
            snapshot is taken.
        snapshot_grad: Full gradient computed at the snapshot point. None
            until first snapshot is taken.
    """

    snapshot: TensorDict | None = None
    snapshot_grad: TensorDict | None = None


SapphireState = SGDState | SAGAState | SVRGState


def get_solver_state(
    op_split: ProxGradSplit | SapphireSplit,
    variable_values: TensorDict,
    config: GradSolverConfig,
) -> GradSolverState:
    """Initialize and return appropriate solver state based on configuration.

    Factory function that creates the correct state object (ProxGradState or
    a SAPPHIRE state variant) based on the provided configuration type.

    Args:
        op_split: Operator splitting object (ProxGradSplit or SapphireSplit).
        variable_values: Initial variable values as a TensorDict.
        config: Solver configuration specifying method and hyperparameters.

    Returns:
        Initialized solver state (ProxGradState, SGDState, SAGAState, or SVRGState).
    """
    if isinstance(config, ProxGradConfig):
        return _build_prox_grad_state(variable_values, config)
    else:
        return _build_sapphire_state(op_split, variable_values, config)


def _build_prox_grad_state(
    variable_values: TensorDict, config: ProxGradConfig
) -> ProxGradState:
    if config.use_acceleration:
        variable_values_prev = variable_values.clone()
    else:
        variable_values_prev = None
    return ProxGradState(
        iter_=0, eta=config.eta, variable_values_prev=variable_values_prev
    )


def _build_sapphire_state(
    op_split: SapphireSplit, variable_values: TensorDict, config: SapphireConfig
) -> SapphireState:
    eta, precond_update_freq = config.eta, config.precond_update_freq
    n = op_split.num_samples

    if config.base_method == "sgd":
        return SGDState(iter_=0, eta=eta, precond_update_freq=precond_update_freq)
    elif config.base_method == "svrg":
        return SVRGState(
            iter_=0,
            eta=eta,
            precond_update_freq=precond_update_freq,
            snapshot=None,
            snapshot_grad=None,
        )
    else:
        device, shape = variable_values["beta"].device, variable_values["beta"].shape
        grad_avg = TensorDict(
            {k: torch.zeros_like(v, device=device) for k, v in variable_values.items()}
        )
        if len(shape) == 2:
            table = torch.zeros((n, shape[1]), device=device)
        else:
            table = torch.zeros(n, device=device)
        return SAGAState(
            iter_=0,
            eta=eta,
            precond_update_freq=precond_update_freq,
            table=table,
            grad_avg=grad_avg,
        )
