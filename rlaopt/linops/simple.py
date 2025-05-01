from typing import Callable

import torch
from torch import vmap

from .base import _BaseLinOp
from rlaopt.utils import _is_callable

__all__ = ["LinOp", "TwoSidedLinOp", "SymmetricLinOp"]


_DEFAULT_DTYPE = torch.get_default_dtype()


class LinOp(_BaseLinOp):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        matmat: Callable | None = None,
        dtype: torch.dtype = _DEFAULT_DTYPE,
    ):
        super().__init__(device=device, shape=shape, dtype=dtype)
        _is_callable(matvec, "matvec")
        if matmat is not None:
            _is_callable(matmat, "matmat")

        self._matvec_fn = matvec

        if matmat is None:
            self._matmat_fn = vmap(self._matvec_fn, in_dims=1, out_dims=1)
        else:
            self._matmat_fn = matmat

    def _matvec(self, x: torch.Tensor):
        return self._matvec_fn(x)

    def _matmat(self, x: torch.Tensor):
        return self._matmat_fn(x)


class TwoSidedLinOp(LinOp):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        rmatvec: Callable,
        matmat: Callable | None = None,
        rmatmat: Callable | None = None,
        dtype: torch.dtype = _DEFAULT_DTYPE,
    ):
        super().__init__(device, shape, matvec, matmat, dtype)

        _is_callable(rmatvec, "rmatvec")
        if rmatmat is not None:
            _is_callable(rmatmat, "rmatmat")

        self._rmatvec_fn = rmatvec
        if rmatmat is None:
            self._rmatmat_fn = vmap(self._rmatvec_fn, in_dims=1, out_dims=1)
        else:
            self._rmatmat_fn = rmatmat

    def _rmatvec(self, x: torch.Tensor):
        return self._rmatvec_fn(x)

    def _rmatmat(self, x: torch.Tensor):
        return self._rmatmat_fn(x)

    @property
    def T(self):
        return TwoSidedLinOp(
            device=self.device,
            shape=torch.Size((self.shape[1], self.shape[0])),
            matvec=self._rmatvec,
            rmatvec=self._matvec,
            matmat=self._rmatmat,
            rmatmat=self._matmat,
        )


class SymmetricLinOp(TwoSidedLinOp):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        matmat: Callable | None = None,
        dtype: torch.dtype = _DEFAULT_DTYPE,
    ):
        super().__init__(device, shape, matvec, matvec, matmat, matmat, dtype)

        if shape[0] != shape[1]:
            raise ValueError(
                f"SymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )

    @property
    def T(self):
        # For symmetric operators, transpose returns self
        return self
