import jax.numpy as jnp
from jax import lax

from sketchyopts.preconditioner import rand_nystrom_approx
from sketchyopts.errors import InputDimError, MatrixNotSquareError

from typing import Optional
from jax.typing import ArrayLike
from jax import Array

KeyArrayLike = ArrayLike


def nystrom_pcg(
    A: ArrayLike,
    b: ArrayLike,
    mu: float,
    rank: int,
    key: KeyArrayLike,
    *,
    x0: Optional[ArrayLike] = None,
    tol: float = 1e-5,
    maxiter: Optional[int] = None,
) -> tuple[Array, Array, int]:
    r"""Nyström preconditioned conjugate gradient (Nyström PCG).

    The function solves the regularized linear system :math:`(A + \mu I)x = b` using Nyström PCG.

    The algorithm uses randomized Nyström preconditioner by implicitly applying

    .. math::
      P^{-1} = (\hat{\lambda}_{l} + \mu) U (\hat{\Lambda} + \mu I)^{-1} U^{T} + (I - U U^{T})

    where :math:`U` and :math:`\hat{\Lambda}` are from rank-:math:`l` randomized Nyström approximation (here :math:`\hat{\lambda}_{l}` is the :math:`l`:sup:`th` diagonal entry of :math:`\hat{\Lambda}`).

    The algorithm terminates if the :math:`\ell_2`-norm of the residual :math:`b - (A + \mu I)\hat{x}` is within the specified tolerance or it has reached the maximal number of iterations.

    References:
      - Z\. Frangella, J. A. Tropp, and M. Udell, `Randomized Nyström preconditioning <https://epubs.siam.org/doi/10.1137/21M1466244>`_. SIAM Journal on Matrix Analysis and Applications, vol. 44, iss. 2, 2023, pp. 718-752.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix.
      b: A vector giving the righthand side of the regularized linear system.
      mu: Regularization parameter (with non-negative value).
      rank: Rank of the randomized Nyström approximation (which coincides with sketch size).
      key: A PRNG key used as the random key.
      x0: Initial guess for the solution (same size as righthand side ``b``; default ``None``). When set to ``None``, the algorithm uses zero vector as starting guess.
      tol: Solution tolerance (default :math:`10^{-5}`).
      maxiter: Maximum number of iterations (default ``None``). When set to ``None``, the algorithm only terminates when the specified tolerance has been achieved. Internally the value gets set to ten times the size of the system.

    Returns:
      A three-element tuple containing

      - **x** – Approximate solution to the regularized linear system.
      - **r** – Residual of the approximate solution.
      - **k** – Total number of iterations to reach to the approximate solution.

    """
    # perform randomized Nyström approximation
    U, S = rand_nystrom_approx(A, rank, key)

    # matrix-vector product for regularized linear operator
    def regularized_A(x):
        return A @ x + mu * x

    # matrix-vector product for inverse Nyström preconditioner
    def inv_preconditioner(x):
        UTx = U.T @ x
        return (S[-1] + mu) * U @ (UTx / (S + mu)) + x - U @ UTx

    # condition evaluation
    def cond_fun(value):
        _, r, _, _, k = value
        r_norm = jnp.linalg.norm(r)
        return (r_norm > tol) & (k < maxiter)

    # PCG iteration
    def body_fun(value):
        x, r, z, p, k = value
        v = regularized_A(p)
        gamma = jnp.dot(r, z)
        alpha = gamma / jnp.dot(p, v)
        x_ = x + alpha * p
        r_ = r - alpha * v
        z_ = inv_preconditioner(r_)
        beta = jnp.dot(r_, z_) / gamma
        p_ = z_ + beta * p
        return x_, r_, z_, p_, k + 1

    # dimension check
    if jnp.ndim(A) != 2:
        raise InputDimError("A", jnp.ndim(A), 2)
    else:
        if jnp.shape(A)[0] != jnp.shape(A)[1]:
            raise MatrixNotSquareError("A", jnp.shape(A))

    b = jnp.squeeze(b)
    if jnp.ndim(b) != 1:
        raise InputDimError("b", jnp.ndim(b), 1)

    # initialization
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if maxiter is None:
        maxiter = jnp.shape(b)[0] * 10  # same behavior as SciPy

    r0 = b - regularized_A(x0)
    p0 = z0 = inv_preconditioner(r0)
    initial_value = (x0, r0, z0, p0, 0)

    x_final, r_final, _, _, k_final = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final, r_final, k_final  # type: ignore
