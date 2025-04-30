from abc import ABC, abstractmethod
from typing import Any

import torch

from rlaopt.utils import _is_torch_size, _is_torch_f32_f64, _is_torch_device


class _BaseLinOp(ABC):
    """Base class for all linear operators."""

    def __init__(self, device: torch.device, shape: torch.Size, dtype: torch.dtype):
        self._check_inputs_base(device, shape, dtype)
        self._device = device
        self._shape = shape
        self._dtype = dtype

    def _check_inputs_base(self, device: Any, shape: Any, dtype: Any):
        _is_torch_device(device, "device")
        _is_torch_size(shape, "shape")
        if len(shape) != 2:
            raise ValueError(f"shape must have two elements. Received {len(shape)}")
        if not all(isinstance(i, int) and i > 0 for i in shape):
            raise ValueError(f"shape must contain positive integers. Received {shape}")
        _is_torch_f32_f64(dtype, "dtype")

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self):
        """Transpose of the linear operator.

        By default, linear operators don't support transposition. Subclasses like
        TwoSidedLinOp should override this property.
        """
        raise NotImplementedError("This linear operator doesn't support transposition")

    @abstractmethod
    def __matmul__(self, x: torch.Tensor):
        """Matrix-vector multiplication."""
        pass

    def __rmatmul__(self, x: torch.Tensor):
        """Right matrix-vector multiplication.

        By default, linear operators don't support right multiplication. Subclasses like
        TwoSidedLinOp should override this method.
        """
        raise NotImplementedError(
            "This linear operator doesn't support right multiplication"
        )
