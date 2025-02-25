from abc import ABC, abstractmethod
from typing import Union

import torch

from rlaopt.models.linops import LinOp


class Preconditioner(ABC):
    def __init__(self, params):
        self._check_inputs(params)
        self.params = params

    @abstractmethod
    def _check_inputs(self, params):
        pass

    @abstractmethod
    def _update(self, A: Union[torch.Tensor, LinOp], *args, **kwargs):
        pass

    @abstractmethod
    def __matmul__(self, x: torch.Tensor):
        pass

    @abstractmethod
    def _inverse_matmul(self, x: torch.Tensor):
        pass

    @property
    def _inv(self):
        return _InvPreconditioner(self)


class _InvPreconditioner:
    def __init__(self, preconditioner: Preconditioner):
        self.preconditioner = preconditioner

    def __matmul__(self, x: torch.Tensor):
        return self.preconditioner._inverse_matmul(x)
