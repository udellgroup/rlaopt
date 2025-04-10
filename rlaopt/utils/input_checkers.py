"""Helper functions for checking input types."""

from typing import Any

import torch


__all__ = [
    "_is_bool",
    "_is_callable",
    "_is_dict",
    "_is_list",
    "_is_set",
    "_is_str",
    "_is_torch_device",
    "_is_torch_f32_f64",
    "_is_torch_size",
    "_is_torch_tensor",
    "_is_torch_tensor_1d_2d",
    "_is_nonneg_float",
    "_is_pos_float",
    "_is_pos_int",
]


def _is_bool(param: Any, param_name: str):
    if not isinstance(param, bool):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type bool"
        )


def _is_callable(param: Any, param_name: str):
    if not callable(param):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type callable"
        )


def _is_dict(param: Any, param_name: str):
    if not isinstance(param, dict):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type dict"
        )


def _is_float(param: Any, param_name: str):
    if not isinstance(param, float):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type float"
        )


def _is_int(param: Any, param_name: str):
    if not isinstance(param, int):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type int"
        )


def _is_list(param: Any, param_name: str):
    if not isinstance(param, list):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type list"
        )


def _is_set(param: Any, param_name: str):
    if not isinstance(param, set):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type set"
        )


def _is_str(param: Any, param_name: str):
    if not isinstance(param, str):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, but expected type str"
        )


def _is_torch_device(param: Any, param_name: str):
    if not isinstance(param, torch.device):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type torch.device"
        )


def _is_torch_dtype(param: Any, param_name: str):
    if not isinstance(param, torch.dtype):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type torch.dtype"
        )


def _is_torch_f32_f64(param: Any, param_name: str):
    _is_torch_dtype(param, param_name)
    if param not in [torch.float32, torch.float64]:
        raise ValueError(
            f"{param_name} is {param}, but expected torch.float32 or torch.float64"
        )


def _is_torch_size(param: Any, param_name: str):
    if not isinstance(param, torch.Size):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type torch.Size"
        )


def _is_torch_tensor(param: Any, param_name: str):
    if not isinstance(param, torch.Tensor):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type torch.Tensor"
        )


def _is_torch_tensor_1d_2d(param: Any, param_name: str):
    _is_torch_tensor(param, param_name)
    if param.ndim not in [1, 2]:
        raise ValueError(
            f"{param_name} must be a 1D or 2D tensor. Received {param.ndim}D tensor."
        )


def _is_nonneg_float(param: Any, param_name: str):
    _is_float(param, param_name)
    if param < 0:
        raise ValueError(f"{param_name} must be non-negative, but received {param}")


def _is_pos_float(param: Any, param_name: str):
    _is_float(param, param_name)
    if param <= 0:
        raise ValueError(f"{param_name} must be positive, but received {param}")


def _is_pos_int(param: Any, param_name: str):
    _is_int(param, param_name)
    if param < 0:
        raise ValueError(f"{param_name} must be positive, but received {param}")
