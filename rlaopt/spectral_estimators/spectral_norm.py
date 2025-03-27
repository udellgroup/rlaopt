"""Functions for estimating spectral norm of matrices/linops."""

from typing import Union, Tuple

import torch

from rlaopt.linops import SymmetricLinOp


__all__ = ["randomized_powering"]


def randomized_powering(
    A: Union[SymmetricLinOp, torch.Tensor], num_iters: int = 10, rtol: float = 10**-3
) -> Tuple[float, torch.Tensor]:

    d = A.shape[0]
    omega = torch.randn(d, device=A.device)
    v = omega / torch.linalg.norm(omega, 2)

    i = 0
    err = torch.inf
    sig = 0.0
    while i < num_iters and err > rtol * sig:
        v_new = A @ v
        sig_new = torch.dot(v, v_new)
        v = v_new / torch.linalg.norm(v_new, 2)
        err = torch.abs(sig_new - sig)
        sig = sig_new
        i += 1
    return sig_new, v
