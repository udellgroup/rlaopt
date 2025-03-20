from typing import Callable, Optional

import torch
from torch import vmap

from rlaopt.utils.input_checkers import (
    _is_callable,
    _is_torch_device,
)
from rlaopt.linops.base_linop import _BaseLinOp


class LinOp(_BaseLinOp):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        matmat: Optional[Callable] = None,
    ):
        super().__init__(shape=shape)
        _is_torch_device(device, "device")
        _is_callable(matvec, "matvec")
        if matmat is not None:
            _is_callable(matmat, "matmat")

        self._device = device
        self._shape = shape
        self._matvec = matvec

        if matmat is None:
            self._matmat = vmap(self._matvec, in_dims=1, out_dims=1)
        else:
            self._matmat = matmat

    @property
    def device(self):
        return self._device

    def __matmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            return self._matvec(x)
        elif x.ndim == 2:
            return self._matmat(x)
        else:
            raise ValueError(f"x must be a 1D or 2D tensor. Received {x.ndim}D tensor.")


class TwoSidedLinOp(LinOp):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        rmatvec: Callable,
        matmat: Optional[Callable] = None,
        rmatmat: Optional[Callable] = None,
    ):
        super().__init__(device, shape, matvec, matmat)

        _is_callable(rmatvec, "rmatvec")
        if rmatmat is not None:
            _is_callable(rmatmat, "rmatmat")

        self._rmatvec = rmatvec
        if rmatmat is None:
            self._rmatmat = vmap(self._rmatvec, in_dims=1, out_dims=1)
        else:
            self._rmatmat = rmatmat

    def __rmatmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)
            result = self._rmatvec(x.T).T
            return result.squeeze(0)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

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
        matmat: Optional[Callable] = None,
    ):
        super().__init__(device, shape, matvec, matvec, matmat, matmat)

        if shape[0] != shape[1]:
            raise ValueError(
                f"SymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )
