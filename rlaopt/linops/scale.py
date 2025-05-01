import torch

from .base import _BaseLinOp
from rlaopt.utils import _is_float

__all__ = ["ScaleLinOp"]


class ScaleLinOp(_BaseLinOp):
    """Linear operator that scales another linear operator by a constant.

    This class implements the decorator pattern and forwards attribute access
    to the underlying operator. Any attribute not found on ScaleLinOp will be
    looked up on the wrapped linear operator.

    Attributes:
        linop: The underlying linear operator.
        scale: The scaling factor.
    """

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
    def linop(self):
        """Return the original linear operator."""
        return self._linop

    @property
    def scale(self):
        """Return the scaling factor."""
        return self._scale

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

    def __getattr__(self, name):
        """Forward attribute access to the underlying linear operator."""
        # This is called only for attributes that don't exist in ScaleLinOp
        try:
            return getattr(self._linop, name)
        except AttributeError:
            raise AttributeError(
                f"Neither '{self.__class__.__name__}' "
                "nor its underlying linear operator "
                f"has attribute '{name}'"
            )

    def __dir__(self):
        """Include attributes from the underlying operator in dir() output."""
        own_attrs = set(super().__dir__())
        linop_attrs = set(dir(self._linop))
        return sorted(own_attrs | linop_attrs)
