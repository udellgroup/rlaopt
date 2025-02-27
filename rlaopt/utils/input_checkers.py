from typing import Any

import torch


def _is_float(param: Any, param_name: str):
    if not isinstance(param, float):
        raise TypeError(
            f"{param_name} is of type {type(param)}, but expected type float"
        )


def _is_int(param: Any, param_name: str):
    if not isinstance(param, int):
        raise TypeError(f"{param_name} is of type {type(param)}, but expected type int")


def _is_str(param: Any, param_name: str):
    if not isinstance(param, str):
        raise TypeError(f"{param_name} is of type {type(param)}, but expected type str")


def _is_torch_device(param: Any, param_name: str):
    if not isinstance(param, torch.device):
        raise TypeError(
            f"{param_name} is of type {type(param)}, but expected type torch.device"
        )


def _is_torch_tensor(param: Any, param_name: str):
    if not isinstance(param, torch.Tensor):
        raise TypeError(
            f"{param_name} is of type {type(param)}, but expected type torch.Tensor"
        )


def _is_nonneg_float(param: Any, param_name: str):
    _is_float(param, param_name)
    if param < 0:
        raise ValueError(f"{param_name} must be non-negative, but received {param}")


def _is_pos_int(param: Any, param_name: str):
    _is_int(param, param_name)
    if param < 0:
        raise ValueError(f"{param_name} must be positive, but received {param}")
