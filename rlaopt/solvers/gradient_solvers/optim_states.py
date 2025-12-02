from dataclasses import dataclass

import torch

from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg.preconditioners.nystrom import Nystrom
from rlaopt.linalg.preconditioners.identity import Identity, IdentityConfig
from rlaopt.solvers.solver_base import SolverState
from rlaopt.splitting.prox_grad_split import ProxGradSplit
from rlaopt.splitting.sapphire_split import SapphireSplit

from .optim_configs import GradSolverConfig, ProxGradConfig, SapphireConfig


@dataclass(frozen=True)
class GradSolverState(SolverState):
    eta: float = 1.0
    err: float = torch.inf
 

@dataclass(frozen=True)
class ProxGradState(GradSolverState):
    variable_values_prev: TensorDict | None = None
    err: float = torch.inf


@dataclass(frozen=True)
class SGDState(GradSolverState):
    P: Nystrom | Identity = Identity(IdentityConfig())
    precond_update_freq: int = 1


@dataclass(frozen=True)
class SAGAState(SGDState):
    table: torch.Tensor | None = None
    grad_avg: TensorDict | None = None
    

@dataclass(frozen=True)
class SVRGState(SGDState):
    snapshot: TensorDict | None = None
    snapshot_grad: TensorDict | None = None
    
SapphireState = SGDState | SAGAState | SVRGState

def get_solver_state(
        op_split: ProxGradSplit | SapphireSplit, variable_values: TensorDict, config: GradSolverConfig
)->GradSolverState:
    if isinstance(config, ProxGradConfig):
        return _build_prox_grad_state(variable_values, config)
    else:
        return _build_sapphire_state(op_split, variable_values, config)


def _build_prox_grad_state(
        variable_values: TensorDict, 
        config: ProxGradConfig
)->ProxGradState:
    if config.use_acceleration == True:
        variable_values_prev = variable_values.clone()
    else:
        variable_values_prev = None
    return ProxGradState(
        iter_=0, eta=config.eta0, variable_values_prev=variable_values_prev
    )


def _build_sapphire_state(
        op_split: SapphireSplit, variable_values: TensorDict, config: SapphireConfig
) -> SapphireState:
    
    eta0, precond_update_freq = config.eta0, config.precond_update_freq
    n = op_split.f.dataloader.dataset.num_samples
    
    if config.base_method == "sgd":
        return SGDState(
            iter_=0, eta = eta0, precond_update_freq=precond_update_freq
        )
    elif config.base_method == "svrg":
        return SVRGState(
            iter_=0, eta = eta0, precond_update_freq=precond_update_freq,
            snapshot=None, snapshot_grad=None
        )
    else:
        grad_avg = TensorDict({
            k: torch.zeros_like(v) for k, v in variable_values.items()
        })
        return SAGAState(
            iter_=0, eta=eta0, precond_update_freq=precond_update_freq,
            table = torch.zeros(n), grad_avg=grad_avg
        )