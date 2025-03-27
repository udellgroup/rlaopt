from typing import Tuple

from rlaopt.linops import TwoSidedLinOp, SymmetricLinOp

from .trace import hutchinson


def fro_norm_est(A: TwoSidedLinOp, k: int, sketch: str) -> Tuple[float]:

    G = SymmetricLinOp(A.device, A.shape, matvec=lambda v: A.T @ (A @ v))

    return hutchinson(G, k, sketch)
