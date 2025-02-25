from typing import Union

import torch

from rlaopt.models.linops import LinOp


class LinSys:
    def __init__(self, A: Union[LinOp, torch.Tensor], b: torch.Tensor):
        self._check_inputs(A, b)
        self.A = A
        self.b = b

    def _check_inputs(self, A: Union[LinOp, torch.Tensor], b: torch.Tensor):
        if not isinstance(A, (LinOp, torch.Tensor)):
            raise ValueError(
                f"A must be an instance of LinOp or a torch.Tensor. \
                             Received {type(A)}"
            )
        if not isinstance(b, torch.Tensor):
            raise ValueError(f"b must be a torch.Tensor. Received {type(b)}")

    def solve():
        raise NotImplementedError
