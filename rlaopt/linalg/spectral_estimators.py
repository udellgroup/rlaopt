"""Module for computing spectral statstics of matrices and linear opeartors."""

from typing import Callable

import torch
from linops import LinearOperator


def randomized_powering(
    A: Callable[[torch.Tensor], torch.Tensor] | LinearOperator | torch.Tensor,
    shape: tuple[int, int] = None,
    tol: float = 1e-3,
    max_iters: int = 10,
    device: torch.device = "cpu",
) -> float:
    """Estimate the largest eigenvalue of a square matrix using the power method.

    Uses the power iteration algorithm with random initialization to compute an
    approximation of the largest eigenvalue (in absolute value) of a symmetric
    linear operator or matrix A.

    Args:
        A: Linear operator representing the matrix. Can be:
            - A callable that takes a torch.Tensor and returns A @ tensor
            - A LinearOperator instance
            - A torch.Tensor matrix
        shape: Shape of the operator as (n, m). Required when A is a Callable,
            otherwise inferred from A.shape. Must be square (n == m).
        tol: Relative convergence tolerance. Iteration stops when the relative
            change in eigenvalue estimate is less than tol. Defaults to 1e-3.
        max_iters: Maximum number of power iterations to perform. Defaults to 10.
        device: PyTorch device for computations. Defaults to "cpu".

    Returns:
        Estimated largest eigenvalue of A.

    Raises:
        ValueError: If A is a Callable and shape is not provided.
        ValueError: If the operator is not square (n != m).

    Examples:
        >>> A = torch.randn(100, 100)
        >>> lambda_max = randomized_powering(A)

        >>> def matvec(v):
        ...     return A @ v
        >>> lambda_max = randomized_powering(matvec, shape=(100, 100))
    """
    if isinstance(A, Callable) and shape is None:
        raise ValueError("When A is of type Callable, shape must be provided.")
    if shape:
        n, m = shape
    else:
        n, m = A.shape

    if n != m:
        raise ValueError(
            f"Input A must have equal dimensions but got input with shape "
            f"A.shape = ({n},{m})."
        )

    if isinstance(A, Callable):

        def matvec_A(y: torch.Tensor) -> torch.Tensor:
            return A(y)
    else:

        def matvec_A(y: torch.Tensor) -> torch.Tensor:
            return A @ y

    y0 = torch.randn(n, device=device)
    y0 /= torch.linalg.norm(y0, ord=2)

    est_lambd_max_old = torch.inf

    for _ in range(max_iters):
        y = matvec_A(y0)
        est_lambd_max = torch.dot(y0, y).item()
        y0 = y / torch.linalg.norm(y, ord=2)
        if abs(est_lambd_max - est_lambd_max_old) < tol * est_lambd_max:
            break
        est_lambd_max_old = est_lambd_max

    return est_lambd_max
