from abc import ABC
from typing import Callable

import torch


class LinOp(ABC):
    def __init__(self, shape: tuple[int, int], matvec: Callable):
        self.shape = shape
        self._matvec = matvec

    def __matmul__(self, x: torch.Tensor):
        return self._matvec(x)


class TwoSidedLinOp(LinOp):
    def __init__(self, shape: tuple[int, int], matvec: Callable, rmatvec: Callable):
        super().__init__(shape, matvec)
        self._rmatvec = rmatvec

    def __rmatmul__(self, x: torch.Tensor):
        return self._rmatvec(x)

    @property
    def T(self):
        return TwoSidedLinOp(
            shape=(self.shape[1], self.shape[0]),
            matvec=self._rmatvec,
            rmatvec=self._matvec,
        )
