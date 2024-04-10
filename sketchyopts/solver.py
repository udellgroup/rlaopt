import jax
import jax.numpy as jnp
from optax._src.base import (
    ScalarOrSchedule,
    GradientTransformation,
    GradientTransformationExtraArgs,
)

from sketchyopts.preconditioner import (
    rand_nystrom_approx,
    update_nystrom_precond,
    scale_by_nystrom_precond,
)
from sketchyopts.util import shareble_state_named_chain, scale_by_ref_learning_rate
from sketchyopts.errors import InputDimError, MatrixNotSquareError

from typing import Optional, Callable, ParamSpec, Concatenate
from jax.typing import ArrayLike
from jax import Array

KeyArrayLike = ArrayLike
P = ParamSpec("P")


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
    r"""The Nyström preconditioned conjugate gradient method (Nyström PCG).

    The function solves the regularized linear system :math:`(A + \mu I)x = b` using
    Nyström PCG.

    Nyström PCG uses randomized Nyström preconditioner by implicitly applying

    .. math::
        P^{-1} = (\hat{\lambda}_{l} + \mu) U (\hat{\Lambda} + \mu I)^{-1} U^{T} + (I - U U^{T})

    where :math:`U` and :math:`\hat{\Lambda}` are from rank-:math:`l` randomized Nyström
    approximation (here :math:`\hat{\lambda}_{l}` is the :math:`l`:sup:`th` diagonal
    entry of :math:`\hat{\Lambda}`).

    Nyström PCG terminates if the :math:`\ell_2`-norm of the residual
    :math:`b - (A + \mu I)\hat{x}` is within the specified tolerance or it has reached
    the maximal number of iterations.

    References:
      - Z\. Frangella, J. A. Tropp, and M. Udell, `Randomized Nyström preconditioning <https://epubs.siam.org/doi/10.1137/21M1466244>`_. SIAM Journal on Matrix Analysis and Applications, vol. 44, iss. 2, 2023, pp. 718-752.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix.
      b: A vector or a two-dimensional array giving the righthand side(s) of the
        regularized linear system.
      mu: Regularization parameter (with non-negative value).
      rank: Rank of the randomized Nyström approximation (which coincides with sketch
        size).
      key: A PRNG key used as the random key.
      x0: Initial guess for the solution (same size as righthand side(s) ``b``; default
        ``None``). When set to ``None``, the algorithm uses zero vector as starting
        guess.
      tol: Solution tolerance (default :math:`10^{-5}`).
      maxiter: Maximum number of iterations (default ``None``). When set to ``None``,
        the algorithm only terminates when the specified tolerance has been achieved.
        Internally the value gets set to ten times the size of the system.

    Returns:
      A four-element tuple containing

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

    # matrix-vector (or mat-mat for multiple righthand sides) product for regularized
    # linear operator
    @jax.jit
    def regularized_A(x):
        return A @ x + mu * x

    # matrix-vector (or mat-mat for multiple righthand sides) product for inverse
    # Nyström preconditioner
    @jax.jit
    def inv_preconditioner(x):
        UTx = U.T @ x
        return (S[-1] + mu) * U @ (UTx / jnp.expand_dims(S + mu, axis=1)) + x - U @ UTx

    # condition evaluation
    def cond_fun(value):
        _, _, _, _, mask, k = value
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
        # perform update on selected columns and ignore padded columns
        # (i.e. with NaN values)
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

    x_final, r_final, _, _, mask_final, k_final = jax.lax.while_loop(
        cond_fun, body_fun, initial_value
    )

    # match solution and residual to the input shape if b has a single dimension
    if b_ndim == 1:
        x_final = jnp.squeeze(x_final)  # type: ignore
        r_final = jnp.squeeze(r_final)

    return x_final, r_final, ~mask_final.astype(bool), k_final  # type: ignore


def sketchysgd(
    rank: int,
    rho: float,
    update_freq: int,
    seed: int,
    f: Callable[Concatenate[Array, P], Array],
    *,
    learning_rate: Optional[float] = None,
) -> GradientTransformation:
    r"""The SketchySGD optimizer.

    SketchySGD is a stochastic quasi-Newton method that uses sketching to approximate the
    curvature of the loss function. It maintains a preconditioner for SGD using
    randomized low-rank Nyström approximations to the subsampled Hessian and
    automatically selects an appropriate learning whenever it updates the preconditioner.

    Example:
      >>> import sketchyopts
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = sketchyopts.sketchysgd(rank=3, rho=1.0, update_freq=0, seed=0, f=f)
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  print('Objective function: {:.2E}'.format(f(params)))
      Objective function: 2.63E-12
      Objective function: 6.37E-25
      Objective function: 1.75E-37
      Objective function: 0.00E+00
      Objective function: 0.00E+00

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates <https://arxiv.org/abs/2211.08597>`_.

    Args:
      rank: Rank of the preconditioner.
      rho: Regularization parameter (with non-negative value).
      update_freq: A non-negative integer that specifies the update frequency of the
        preconditioner. When set to ``0`` or :math:`\infty` (*e.g.* ``jax.numpy.inf`` or
        ``numpy.inf``), the optimizer uses constant preconditioner that is constructed at
        the beginning of the optimization process.
      seed: An integer used as a seed to generate random numbers.
      f: A scalar-valued loss/objective function to be optimized. The function needs to
        have the optimization variable as its first argument.
      learning_rate: Step size for applying updates (default ``None``). It can either be
        a fixed scalar value or a schedule based on step count. When set to ``None``, the
        algorithm adaptively chooses a learning rate whenever the preconditioner is updated.

    Returns:
      The corresponding `GradientTransformation <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformation>`_ object.

    """
    return shareble_state_named_chain(
        (
            "update_precond",
            update_nystrom_precond(
                rank, rho, update_freq, seed, f, adaptive_lr=learning_rate is None
            ),
        ),
        ("scale_by_precond", scale_by_nystrom_precond(rho, ref_state="update_precond")),
        (
            "scale_by_lr",
            scale_by_ref_learning_rate(learning_rate, ref_state="update_precond"),
        ),
    )
