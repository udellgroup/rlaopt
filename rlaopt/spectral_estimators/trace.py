"""Module containing Hutchinson-type methods for trace estimation."""

import math
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
    Omega = get_sketch(sketch, "left", k, A.shape[0], A.dtype, A.device)
    Omega_A = Omega._apply_left(A)
    Omega_A_Omega_T = Omega._apply_right_trans(Omega_A)
    d = Omega_A_Omega_T.diag()
    trace = torch.mean(d)
    return trace.item()


def hutch_plus_plus(
    A: SymmetricLinOp, k: int, sketch: str, sketch_fraction: float = 1 / 3
) -> float:
    """Estimate the trace of a symmetric linear operator using the Hutch++ estimator.

    Hutch++ combines sketching and the Hutchinson approach to improve trace estimation.

    Args:
        A (SymmetricLinOp): The symmetric linear operator.
        k (int): Number of probe vectors (sketch size).
        sketch (str): Identifier for the random sketching method.


    Returns:
        float: Estimated trace of A.
    """
    if sketch_fraction >= 1 / 2:
        raise ValueError("sketch_fraction must be smaller than 1/2!!")

    s_dim = math.ceil(k * sketch_fraction)
    g_dim = k - 2 * s_dim

    trace_sketch = torch.tensor(0.0, dtype=A.dtype, device=A.device)
    trace_hutch = torch.tensor(0.0, dtype=A.dtype, device=A.device)

    if s_dim > 0:
        S = get_sketch(sketch, "right", s_dim, A.shape[0], A.dtype, A.device)
        AS = S._apply_right(A)
        Q, _ = torch.linalg.qr(AS, mode="reduced")  # Economic QR
        AQ = A @ Q
        trace_sketch = torch.trace(Q.T @ AQ)
    else:
        Q = None  # For safe code path

    if g_dim > 0:
        G = get_sketch(sketch, "right", g_dim, A.shape[0], A.dtype, A.device)
        if Q is not None:
            QT_G = G._apply_right(Q.T)
            Q_QT_G = Q @ QT_G
            G_proj = G.Omega_mat - Q_QT_G
        else:
            G_proj = G.Omega_mat

        AG = A @ G_proj
        trace_hutch = torch.trace(G_proj.T @ AG) / g_dim
    # else: Keep trace_hutch = 0.0

    return (trace_sketch + trace_hutch).item()
