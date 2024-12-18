from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from sketchyopts.base import LinearSolveState, SolverState
from sketchyopts.low_rank_approx import rand_nystrom_approx
from sketchyopts.util import tree_size, tree_zeros_like

from .cg import abstract_cg

KeyArrayLike = ArrayLike


def nystrom_pcg(
    A: Any,
    b: ArrayLike,
    mu: float,
    rank: int,
    key: KeyArrayLike,
    *,
    x0: Optional[ArrayLike] = None,
    maxiter: Optional[int] = None,
    tol: float = 1e-5,
) -> SolverState:
    r"""The Nyström preconditioned conjugate gradient method (Nyström PCG).

    The function solves the regularized linear system :math:`(A + \mu I)x = b` using
    Nyström PCG.

    Nyström PCG uses randomized Nyström preconditioner by implicitly applying

    .. math::
        P^{-1}
        = (\hat{\lambda}_{l} + \mu) U (\hat{\Lambda} + \mu I)^{-1} U^{T} + (I - U U^{T})

    where :math:`U` and :math:`\hat{\Lambda}` are from rank-:math:`l` randomized Nyström
    approximation (here :math:`\hat{\lambda}_{l}` is the :math:`l`:sup:`th` diagonal
    entry of :math:`\hat{\Lambda}`).

    Nyström PCG terminates if the :math:`\ell^2`-norm of the residual
    :math:`b - (A + \mu I)\hat{x}` is within the specified tolerance or it has reached
    the maximal number of iterations.

    References:
      - Z\. Frangella, J. A. Tropp, and M. Udell, `Randomized Nyström preconditioning
        <https://epubs.siam.org/doi/10.1137/21M1466244>`_. SIAM Journal on Matrix
        Analysis and Applications, vol. 44, iss. 2, 2023, pp. 718-752.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix.
      b: A vector or a two-dimensional array giving the righthand side of the
        regularized linear system.
      mu: Regularization parameter. Expect a non-negative value.
      rank: Rank of the randomized Nyström approximation (which coincides with sketch
        size). Expect a positive value.
      key: A PRNG key used as the random key.
      x0: Initial guess for the solution (same size as righthand side ``b``; default
        ``None``). When set to ``None``, the algorithm uses zero vector as starting
        guess.
      maxiter: Maximum number of iterations (default ``None``). When set to ``None``,
        the algorithm only terminates when the specified tolerance has been achieved.
        Internally the value gets set to :math:`10` times the size of the system (same
        as Scipy CG implementation).
      tol: Solution tolerance (default ``1e-5``).

    Returns:
      A two-element tuple containing

      - **x** – Approximate solution to the regularized linear system. Solution has the
        same size as righthand side(s) ``b``.
      - **r** – Residual of the approximate solution. Residual has the same size as
        righthand side(s) ``b``.
      - **status** – Whether or not the approximate solution has converged for each
        righthand side. Status has the same size as the number of righthand side(s).
      - **k** – Total number of iterations to reach to the approximate solution.
    """
    # perform randomized Nyström approximation
    U, S = rand_nystrom_approx(A, rank, key)

    # matrix-vector product for inverse Nyström preconditioner
    @jax.jit
    def M(x):
        UTx = U.T @ x
        return (S[-1] + mu) * U @ (UTx / jnp.expand_dims(S + mu, axis=1)) + x - U @ UTx

    # initialize the preconditioned conjugate gradient method
    if x0 is None:
        x0 = tree_zeros_like(b)
    if maxiter is None:
        maxiter = int(tree_size(b) * 10)

    # perform the preconditioned conjugate gradient method
    solution, residual, _, _, num_steps = abstract_cg(A, b, mu, x0, tol, maxiter, M)

    # print out warning message if needed
    if num_steps == maxiter:
        print(
            "Warning: solver did not converge within the maximum number of iterations."
        )

    return SolverState(
        params=solution,
        state=LinearSolveState(iter_num=num_steps, maxiter=maxiter, residual=residual),
    )
