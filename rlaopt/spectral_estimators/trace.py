"""Module containing Hutchinson-type methods for trace estimation."""

import torch

from rlaopt.linops import SymmetricLinOp
from rlaopt.sketches import get_sketch


__all__ = ["hutchinson", "hutch_plus_plus"]


def hutchinson(A: SymmetricLinOp, k: int, sketch: str) -> float:
    """Estimate the trace of a symmetric linear operator using the Hutchinson estimator.

    Args:
        A (SymmetricLinOp): The symmetric linear operator.
        k (int): Number of probe vectors (sketch size).
        sketch (str): Identifier for the random sketching method.

    Returns:
        float: Estimated trace of A.
    """
    Omega = get_sketch(sketch, "left", k, A.shape[0], torch.float32, A.device)
    Omega_A = Omega._apply_left(A)
    Omega_A_Omega_T = Omega._apply_right_trans(Omega_A)
    d = Omega_A_Omega_T.diag()
    trace = torch.sum(d) / k
    # var = 1 / (k - 1) * torch.sum(k * d - trace)
    return trace.item()


def hutch_plus_plus(A: SymmetricLinOp, k: int, sketch: str) -> float:
    """Estimate the trace of a symmetric linear operator using the Hutch++ estimator.

    Hutch++ combines sketching and the Hutchinson approach to improve trace estimation.

    Args:
        A (SymmetricLinOp): The symmetric linear operator.
        k (int): Number of probe vectors (sketch size).
        sketch (str): Identifier for the random sketching method.

    Returns:
        float: Estimated trace of A.
    """
    s_dim = k // 2
    g_dim = k - s_dim
    S = get_sketch(sketch, "right", k, A.shape[0], torch.float32, A.device)
    AS = S._apply_right(A)
    Q, _ = torch.linalg.qr(AS, mode="reduced")  # Economic QR

    G = get_sketch(sketch, "right", k, A.shape[0], torch.float32, A.device)
    QT_G = G._apply_right(Q.T)
    Q_QT_G = Q @ QT_G
    G_proj = G.Omega_mat - Q_QT_G

    AQ = A @ Q
    trace_sketch = torch.trace(Q.T @ AQ)
    AG = A @ G_proj
    trace_hutch = torch.trace(G_proj.T @ AG) / g_dim

    return (trace_sketch + trace_hutch).item()
