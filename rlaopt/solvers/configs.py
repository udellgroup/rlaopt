from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any

import torch

from rlaopt.preconditioners import (
    PreconditionerConfig,
    IdentityConfig,
    _is_precond_config,
)
from rlaopt.utils import (
    _is_bool,
    _is_nonneg_float,
    _is_pos_float,
    _is_pos_int,
    _is_torch_device,
)


__all__ = [
    "SAPAccelConfig",
    "SolverConfig",
    "PCGConfig",
    "SAPConfig",
    "_is_solver_config",
    "_get_solver_name",
]


@dataclass(kw_only=True, frozen=False)
class SAPAccelConfig:
    mu: float
    nu: float

    def __post_init__(self):
        _is_pos_float(self.mu, "mu")
        _is_pos_float(self.nu, "nu")
        if self.mu > self.nu:
            raise ValueError("mu must be less than or equal to nu")
        if self.mu * self.nu > 1:
            raise ValueError("mu * nu must be less than or equal to 1")


def _is_sap_accel_config(param: Any, param_name: str):
    if not isinstance(param, SAPAccelConfig):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type SAPAccelConfig"
        )


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


@dataclass(kw_only=True, frozen=False)
class SAPConfig(SolverConfig):
    device: torch.device
    max_iters: int = 1000
    atol: float = 0.0
    rtol: float = 1e-5
    precond_config: PreconditionerConfig = IdentityConfig()
    blk_sz: int
    accel: bool = True
    accel_config: SAPAccelConfig = (None,)
    power_iters: int = 10

    def __post_init__(self):
        _is_torch_device(self.device, "device")
        _is_pos_int(self.max_iters, "max_iters")
        _is_nonneg_float(self.atol, "atol")
        _is_nonneg_float(self.rtol, "rtol")
        _is_precond_config(self.precond_config, "precond_config")
        _is_pos_int(self.blk_sz, "blk_sz")
        _is_bool(self.accel, "accel")
        if self.accel:
            if self.accel_config is None:
                raise ValueError("accel_config must be specified if accel is True")
            _is_sap_accel_config(self.accel_config, "accel_config")
        _is_pos_int(self.power_iters, "power_iters")


def _is_solver_config(param: Any, param_name: str):
    if not isinstance(param, SolverConfig):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type SolverConfig"
        )


CONFIG_TO_NAME = {
    PCGConfig: "pcg",
    SAPConfig: "sap",
}


def _get_solver_name(solver_config: SolverConfig) -> str:
    config_class = solver_config.__class__
    return CONFIG_TO_NAME.get(config_class)
