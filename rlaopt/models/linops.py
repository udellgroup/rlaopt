from abc import ABC
from typing import Callable

import torch
from torch import vmap


class LinOp(ABC):
    def __init__(
        self, shape: tuple[int, int], matvec: Callable, matmat: Callable = None
    ):
        self._check_shape(shape)
        self._check_callable(matvec, "matvec")
        if matmat is not None:
            self._check_callable(matmat, "matmat")

        self.shape = shape
        self._matvec = matvec

        if matmat is None:
            self._matmat = vmap(self._matvec, in_dims=1, out_dims=1)
        else:
            self._matmat = matmat

    def _check_shape(self, shape: tuple[int, int]):
        if not isinstance(shape, tuple):
            raise ValueError(f"shape must be a tuple. Received {type(shape)}")
        if len(shape) != 2:
            raise ValueError(f"shape must have two elements. Received {len(shape)}")
        if not all(isinstance(i, int) and i > 0 for i in shape):
            raise ValueError(f"shape must contain positive integers. Received {shape}")

    def _check_callable(self, func: Callable, name: str):
        if not callable(func):
            raise ValueError(f"{name} must be a callable. Received {type(func)}")

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
        shape: tuple[int, int],
        matvec: Callable,
        rmatvec: Callable,
        matmat: Callable = None,
        rmatmat: Callable = None,
    ):
        # TODO(pratik): eliminate redundancy in the checks
        self._check_shape(shape)
        self._check_callable(matvec, "matvec")
        self._check_callable(rmatvec, "rmatvec")
        if matmat is not None:
            self._check_callable(matmat, "matmat")
        if rmatmat is not None:
            self._check_callable(rmatmat, "rmatmat")

        super().__init__(shape, matvec, matmat)

        self._rmatvec = rmatvec
        if rmatmat is None:
            self._rmatmat = vmap(self._rmatvec, in_dims=1, out_dims=1)
        else:
            self._rmatmat = rmatmat

    def __rmatmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(1)
            result = self._rmatvec(x.T).T
            return result.squeeze(1)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

    @property
    def T(self):
        return TwoSidedLinOp(
            shape=(self.shape[1], self.shape[0]),
            matvec=self._rmatvec,
            rmatvec=self._matvec,
        )


class SymmetricLinOp(TwoSidedLinOp):
    def __init__(self, shape: tuple[int, int], matvec: Callable):
        if shape[0] != shape[1]:
            raise ValueError(
                f"SymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )
        super().__init__(shape, matvec, matvec)
