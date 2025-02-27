from abc import ABC
from dataclasses import dataclass

import torch

from rlaopt.preconditioners import (
    PreconditionerConfig,
    IdentityConfig,
    _is_precond_config,
)
from rlaopt.utils import _is_nonneg_float, _is_pos_int, _is_torch_device


@dataclass(kw_only=True, frozen=False)
class SolverConfig(ABC):
    pass


@dataclass(kw_only=True, frozen=False)
class PCGConfig(SolverConfig):
    device: torch.device
    max_iters: int = 1000
    atol: float = 0.0
    rtol: float = 1e-5
    precond_config: PreconditionerConfig = IdentityConfig()

    def __post_init__(self):
        _is_torch_device(self.device, "device")
        _is_pos_int(self.max_iters, "max_iters")
        _is_nonneg_float(self.atol, "atol")
        _is_nonneg_float(self.rtol, "rtol")
        _is_precond_config(self.precond_config, "precond_config")
