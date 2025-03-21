from abc import ABC, abstractmethod
from typing import Any

import torch

from rlaopt.utils import _is_torch_size


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

    @abstractmethod
    def __matmul__(self, x: torch.Tensor):
        """Matrix-vector multiplication."""
        pass
