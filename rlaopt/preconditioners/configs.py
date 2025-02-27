from abc import ABC
from dataclasses import dataclass
from typing import Any

import torch

from rlaopt.utils import _is_torch_device, _is_nonneg_float, _is_pos_int


@dataclass(kw_only=True, frozen=False)
class PreconditionerConfig(ABC):
    pass


@dataclass(kw_only=True, frozen=False)
class IdentityConfig(PreconditionerConfig):
    pass


@dataclass(kw_only=True, frozen=False)
class NewtonConfig(PreconditionerConfig):
    rho: float
    device: torch.device

    def __post_init__(self):
        _is_nonneg_float(self.rho, "rho")
        _is_torch_device(self.device, "device")


@dataclass(kw_only=True, frozen=False)
class NystromConfig(PreconditionerConfig):
    rank: int
    rho: float
    device: torch.device
    type: str = "gauss"

    def __post_init__(self):
        _is_pos_int(self.rank, "rank")
        _is_nonneg_float(self.rho, "rho")
        _is_torch_device(self.device, "device")


def _is_precond_config(param: Any, param_name: str):
    if not isinstance(param, PreconditionerConfig):
        raise TypeError(
            f"{param_name} is of type {type(param)}, but expected \
                type PreconditionerConfig"
        )
