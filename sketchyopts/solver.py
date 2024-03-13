import jax.numpy as jnp
from jax import lax
from jax import jit

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
) -> tuple[Array, Array, Array, int]:
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
      b: A vector or a two-dimensional array giving the righthand side(s) of the regularized linear system.
      mu: Regularization parameter (with non-negative value).
      rank: Rank of the randomized Nyström approximation (which coincides with sketch size).
      key: A PRNG key used as the random key.
      x0: Initial guess for the solution (same size as righthand side(s) ``b``; default ``None``). When set to ``None``, the algorithm uses zero vector as starting guess.
      tol: Solution tolerance (default :math:`10^{-5}`).
      maxiter: Maximum number of iterations (default ``None``). When set to ``None``, the algorithm only terminates when the specified tolerance has been achieved. Internally the value gets set to ten times the size of the system.

    Returns:
      A four-element tuple containing

      - **x** – Approximate solution to the regularized linear system. Solution has the same size as righthand side(s) ``b``.
      - **r** – Residual of the approximate solution. Residual has the same size as righthand side(s) ``b``.
      - **status** – Whether or not the approximate solution has converged for each righthand side. Status has the same size as the number of righthand side(s).
      - **k** – Total number of iterations to reach to the approximate solution.

    """
    # perform randomized Nyström approximation
    U, S = rand_nystrom_approx(A, rank, key)

    # matrix-vector (or mat-mat for multiple righthand sides) product for regularized linear operator
    @jit
    def regularized_A(x):
        return A @ x + mu * x

    # matrix-vector (or mat-mat for multiple righthand sides) product for inverse Nyström preconditioner
    @jit
    def inv_preconditioner(x):
        UTx = U.T @ x
        return (S[-1] + mu) * U @ (UTx / jnp.expand_dims(S + mu, axis=1)) + x - U @ UTx

    # condition evaluation
    def cond_fun(value):
        (
            _,
            _,
            _,
            _,
            mask,
            k,
        ) = value
        return (jnp.sum(mask) > 0) & (k < maxiter)

    # PCG iteration
    def body_fun(value):
        x, r, z, p, mask, k = value
        # select only columns corresponding to the righthand side that has yet converged
        # populate the remaining columns with NaN
        xs = jnp.where(mask > 0, x[:,], jnp.nan)
        rs = jnp.where(mask > 0, r[:,], jnp.nan)
        zs = jnp.where(mask > 0, z[:,], jnp.nan)
        ps = jnp.where(mask > 0, p[:,], jnp.nan)
        # perform update on selected columns and ignore padded columns (i.e. with NaN values)
        v = regularized_A(ps)
        gamma = jnp.sum(rs * zs, axis=0, keepdims=True)  # type: ignore
        alpha = gamma / jnp.sum(ps * v, axis=0, keepdims=True)
        x_ = jnp.where(mask > 0, (xs + alpha * ps)[:,], x[:,])
        r_s = rs - alpha * v
        r_ = jnp.where(mask > 0, r_s[:,], r[:,])
        z_s = inv_preconditioner(r_s)
        z_ = jnp.where(mask > 0, z_s[:,], z[:,])
        beta = jnp.sum(r_s * z_s, axis=0, keepdims=True) / gamma
        p_ = jnp.where(mask > 0, (z_s + beta * ps)[:,], p[:,])
        r_norm = jnp.linalg.norm(r_s, axis=0)
        mask_ = jnp.where(r_norm > tol, 1, 0)  # NaN always evaluates to False
        return x_, r_, z_, p_, mask_, k + 1

    # dimension check
    if jnp.ndim(A) != 2:
        raise InputDimError("A", jnp.ndim(A), 2)
    else:
        if jnp.shape(A)[0] != jnp.shape(A)[1]:
            raise MatrixNotSquareError("A", jnp.shape(A))

    if jnp.ndim(b) not in [1, 2]:
        raise InputDimError("b", jnp.ndim(b), [1, 2])

    # initialization
    b_ndim = jnp.ndim(b)
    if b_ndim == 1:
        b = jnp.expand_dims(b, axis=1)
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if maxiter is None:
        maxiter = jnp.shape(b)[0] * 10  # same behavior as SciPy

    # initial step
    r0 = b - regularized_A(x0)
    p0 = z0 = inv_preconditioner(r0)
    mask0 = jnp.ones(jnp.shape(b)[1], dtype=int)
    initial_value = (x0, r0, z0, p0, mask0, 0)

    x_final, r_final, _, _, mask_final, k_final = lax.while_loop(
        cond_fun, body_fun, initial_value
    )

    # match solution and residual to the input shape if b has a single dimension
    if b_ndim == 1:
        x_final = jnp.squeeze(x_final)  # type: ignore
        r_final = jnp.squeeze(r_final)

    return x_final, r_final, ~mask_final.astype(bool), k_final  # type: ignore
