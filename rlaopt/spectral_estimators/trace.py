"""Module containing Hutchinson type methods for trace estimation."""

from typing import Tuple, Union

import torch

from rlaopt.linops import SymmetricLinOp
from rlaopt.sketches import get_sketch


def hutchinson(A: SymmetricLinOp, k: int, sketch: str) -> Tuple[float, float]:

    Omega = get_sketch(sketch, "left", k, A.shape[0], A.device)
    Omega_A = Omega._apply_left(A)
    Omega_A_Omega_T = Omega._apply_right_trans(Omega_A)
    d = Omega_A_Omega_T.diag()
    trace = torch.sum(d)
    var = 1 / (k - 1) * torch.sum(k * d - trace)
    return trace, var


def hutch_plus_plus(
    A: Union[SymmetricLinOp, torch.Tensor], k: int, sketch: str
) -> float:
    pass
