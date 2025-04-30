import torch

from .base import _BaseLinOp
from rlaopt.utils import _is_float


class ScaleLinOp(_BaseLinOp):
    """Linear operator that scales another linear operator by a constant."""

    def __init__(
        self,
        linop: _BaseLinOp,
        scale: float,
    ):
        """Initialize a scaled linear operator.

        Parameters
        ----------
        linop : _BaseLinOp
            The linear operator to scale
        scale : float
            The scaling factor to apply
        """
        # Initialize with the shape and dtype from the original operator
        super().__init__(shape=linop.shape, dtype=linop.dtype)
        _is_float(scale, "scale")

        self._linop = linop
        self._scale = scale

    @property
    def device(self):
        """Return the device of the underlying linear operator."""
        return self._linop.device

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
