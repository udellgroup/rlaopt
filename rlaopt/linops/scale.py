import torch

from .base import _BaseLinOp
from rlaopt.utils import _is_float

__all__ = ["ScaleLinOp"]


class ScaleLinOp(_BaseLinOp):
    """Linear operator that scales another linear operator by a constant."""

    def __init__(
        self,
        linop: _BaseLinOp,
        scale: float,
    ):
        # Initialize with the shape and dtype from the original operator
        super().__init__(device=linop.device, shape=linop.shape, dtype=linop.dtype)
        _is_float(scale, "scale")

        self._linop = linop
        self._scale = scale

    @property
    def scale(self):
        """Return the scaling factor."""
        return self._scale

    @property
    def linop(self):
        """Return the original linear operator."""
        return self._linop

    def __matmul__(self, x: torch.Tensor):
        """Apply scaled matrix-vector multiplication."""
        return self._scale * (self._linop @ x)

    def __rmatmul__(self, x: torch.Tensor):
        """Apply scaled right matrix-vector multiplication."""
        return self._scale * (x @ self._linop)

    @property
    def T(self):
        """Return the transpose of the scaled linear operator."""
        return ScaleLinOp(self._linop.T, self._scale)
