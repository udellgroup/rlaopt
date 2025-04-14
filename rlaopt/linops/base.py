from abc import ABC, abstractmethod
from typing import Any

import torch

from rlaopt.utils import _is_torch_size, _is_torch_f32_f64


class _BaseLinOp(ABC):
    """Base class for all linear operators."""

    def __init__(self, shape: torch.Size, dtype: torch.dtype):
        self._check_inputs_base(shape, dtype)
        self._shape = shape
        self._dtype = dtype

    def _check_inputs_base(self, shape: Any, dtype: Any):
        _is_torch_size(shape, "shape")
        if len(shape) != 2:
            raise ValueError(f"shape must have two elements. Received {len(shape)}")
        if not all(isinstance(i, int) and i > 0 for i in shape):
            raise ValueError(f"shape must contain positive integers. Received {shape}")
        _is_torch_f32_f64(dtype, "dtype")

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @abstractmethod
    def __matmul__(self, x: torch.Tensor):
        """Matrix-vector multiplication."""
        pass
