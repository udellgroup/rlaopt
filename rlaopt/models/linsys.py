from typing import Union, Optional

import torch

from rlaopt.models.linops import LinOp


class LinSys:
    def __init__(
        self, A: Union[LinOp, torch.Tensor], b: torch.Tensor, reg: Optional[float] = 0.0
    ):
        self._check_inputs(A, b, reg)
        self._A = A
        self._b = b
        self._reg = reg

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def reg(self):
        return self._reg

    def _check_inputs(self, A: Union[LinOp, torch.Tensor], b: torch.Tensor, reg: float):
        if not isinstance(A, (LinOp, torch.Tensor)):
            raise ValueError(
                f"A must be an instance of LinOp or a torch.Tensor. \
                             Received {type(A)}"
            )
        if not isinstance(b, torch.Tensor):
            raise ValueError(f"b must be a torch.Tensor. Received {type(b)}")
        if not isinstance(reg, float) or reg < 0:
            raise ValueError("reg must be a non-negative float")

    def solve():
        raise NotImplementedError
