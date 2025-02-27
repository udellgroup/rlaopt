from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any

import torch

from rlaopt.preconditioners import (
    PreconditionerConfig,
    IdentityConfig,
    _is_precond_config,
)
from rlaopt.utils import _is_nonneg_float, _is_pos_int, _is_torch_device


@dataclass(kw_only=True, frozen=False)
class SolverConfig(ABC):
    def to_dict(self) -> dict:
        data_dict = asdict(self)
        for key, value in data_dict.items():
            if isinstance(value, torch.device):
                data_dict[key] = str(value)  # Convert torch.device to string
            elif isinstance(value, PreconditionerConfig):
                data_dict[
                    key
                ] = value.to_dict()  # Ensure nested configs are also converted
        return data_dict


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


def _is_solver_config(param: Any, param_name: str):
    if not isinstance(param, SolverConfig):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type SolverConfig"
        )
