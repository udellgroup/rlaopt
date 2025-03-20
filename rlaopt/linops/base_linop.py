from abc import ABC
from typing import Any

import torch

from rlaopt.utils import _is_torch_size


def _is_linop_or_torch_tensor(param: Any, param_name: str):
    if not isinstance(param, (_BaseLinOp, torch.Tensor)):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type _BaseLinOp or torch.Tensor"
        )


class _BaseLinOp(ABC):
    """Base class for all linear operators."""

    def __init__(self, shape: torch.Size):
        self._check_shape(shape)
        self._shape = shape

    def _check_shape(self, shape: Any):
        _is_torch_size(shape, "shape")
        if len(shape) != 2:
            raise ValueError(f"shape must have two elements. Received {len(shape)}")
        if not all(isinstance(i, int) and i > 0 for i in shape):
            raise ValueError(f"shape must contain positive integers. Received {shape}")

    @property
    def shape(self):
        return self._shape
