from rlaopt.linops import TwoSidedLinOp, SymmetricLinOp

from .trace import hutchinson


__all__ = ["fro_norm_est"]


def fro_norm_est(A: TwoSidedLinOp, k: int, sketch: str) -> tuple[float]:

    G = SymmetricLinOp(A.device, A.shape, matvec=lambda v: A.T @ (A @ v))

    return hutchinson(G, k, sketch)
