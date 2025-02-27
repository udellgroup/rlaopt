from abc import ABC, abstractmethod
from typing import Union

import torch

from rlaopt.utils import LinOp
from rlaopt.preconditioners.configs import PreconditionerConfig


class Preconditioner(ABC):
    def __init__(self, config: PreconditionerConfig):
        self.config = config

    @abstractmethod
    def _update(
        self,
        A: Union[torch.Tensor, LinOp],
        device: torch.device,
        *args: list,
        **kwargs: dict
    ):
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
