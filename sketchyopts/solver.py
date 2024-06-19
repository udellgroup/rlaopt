import jax
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree
import numpy as np

from sketchyopts.preconditioner import rand_nystrom_approx
from sketchyopts.base import SolverState, PromiseSolver
from sketchyopts.util import generate_random_batch
from sketchyopts.tree_util import tree_add_scalar_mul, tree_l2_norm
from sketchyopts.errors import InputDimError, MatrixNotSquareError

from dataclasses import dataclass
from typing import Optional, Union, Callable, NamedTuple, Any
from jax.typing import ArrayLike
from jax import Array

KeyArray = Array
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


class SketchySGDState(NamedTuple):
    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: Callable
    lr: Array


@dataclass(eq=False, kw_only=True)
class SketchySGD(PromiseSolver):
    r"""The SketchySGD optimizer.

    SketchySGD is a stochastic quasi-Newton method that uses sketching to approximate the
    curvature of the loss function. It maintains a preconditioner for SGD using
    randomized low-rank Nyström approximations to the subsampled Hessian and
    automatically selects an appropriate learning whenever it updates the preconditioner.

    Example:
      .. highlight:: python
      .. code-block:: python

        from sketchyopts.solver import SketchySGD

        def ridge_reg_objective(params, l2reg, data):
            X, y = data[:,:sample_dim], data[:,sample_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.dot(w ** 2)

        solver = SketchySGD(ridge_reg_objective, ...)
        solver.run(init_params, data, l2reg=l2reg)

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates <https://arxiv.org/abs/2211.08597>`_.

    Args:
      fun: Objective function. The function needs to have the optimization variable as its first argument.
      grad_fun: Optional gradient oracle that is expected to have the same function signature as the objective.
      hvp_fun: Optional hessian-vector-product oracle that takes optimization variable ``params`` as its first argument, vector ``v`` as its second arguments.
      pre_update: Optional gradient update function that gets called before every parameter update.
      rank: Rank of the preconditioner.
      rho: Regularization parameter (with non-negative value).
      update_freq: A non-negative integer that specifies the update frequency of the
        preconditioner. When set to ``0`` or :math:`\infty` (*e.g.* ``jax.numpy.inf`` or
        ``numpy.inf``), the optimizer uses constant preconditioner that is constructed at
        the beginning of the optimization process.
      seed: An integer used as a seed to generate random numbers.
      learning_rate: Step size for applying updates (default :math:`0.5`). It can either be
        a fixed scalar value or a schedule based on step count. If a fixed scalar value is
        provided, the algorithm uses the value as the scaling factor to the adaptively chosen
        learning rate whenever the preconditioner is updated.
      maxiter: Maximum number of iterations to run the solver (default :math:`20`).
      tol: Threshold of the gradient norm used for terminating the solver (default :math:`1e-3`).
      verbose:

    """

    update_freq: int

    def _init_state(self) -> SketchySGDState:
        return SketchySGDState(
            iter_num=jnp.asarray(0),
            value=jnp.asarray(jnp.inf),
            error=jnp.asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=lambda g: g,
            lr=jnp.asarray(0.0),
        )

    def _update_params(self, params, state, data, *args, **kwargs) -> SolverState:
        batch_idx, key = generate_random_batch(data, self.grad_batch_size, state.key)

        if self.pre_update:
            solver_state = self.pre_update(
                params, state, *args, **kwargs, data=data[batch_idx]
            )

        value, grad = self._value_and_grad(
            params, *args, **kwargs, data=data[batch_idx]
        )
        flattend_grad, unravel_fn = ravel_pytree(grad)
        direction = unravel_fn(state.precond(flattend_grad))
        params = tree_add_scalar_mul(params, -state.lr, direction)
        error = tree_l2_norm(grad, squared=False)

        return SolverState(
            params=params,
            state=SketchySGDState(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
                precond=state.precond,
                lr=state.lr,
            ),
        )

    def run(self, init_params: Any, data: ArrayLike, *args, **kwargs) -> SolverState:
        """Runs the optimization loop over an iterator.

        Both iterators should not exhaust before the run ends.

        Args:
          init_params: pytree containing the initial parameters.
          gradient_iterator: iterator generating random data batches of size :math:`b_g` for updating the parameters
          hessian_iterator: iterator generating random data batches of size :math:`b_h` for updating the preconditioner
          *args: additional positional arguments to be passed to ``fun`` (or ``grad_fun`` if provided).
          **kwargs: additional keyword arguments to be passed to ``fun`` (or ``grad_fun`` if provided).
        Returns:
          (params, state)
        """

        params = init_params
        state = self._init_state()

        for i in range(self.maxiter):

            if (self.update_freq == 0 and state.iter_num == 0) or (
                self.update_freq > 0 and state.iter_num % self.update_freq == 0
            ):
                params, state = self._update_precond(
                    params, state, data, *args, **kwargs
                )

            params, state = self._update_params(params, state, data, *args, **kwargs)

            if state.error < self.tol:
                jax.debug.print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)


class SketchySVRGState(NamedTuple):
    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: Callable
    lr: Array


@dataclass(eq=False, kw_only=True)
class SketchySVRG(PromiseSolver):
    r"""The SketchySVRG optimizer.

    [SketchySVRG description]

    Example:
      .. highlight:: python
      .. code-block:: python

        from sketchyopts.solver import SketchySVRG

        def ridge_reg_objective(params, l2reg, data):
            X, y = data[:,:sample_dim], data[:,sample_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.dot(w ** 2)

        solver = SketchySVRG(ridge_reg_objective, ...)
        solver.run(init_params, data, l2reg=l2reg)

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates <https://arxiv.org/abs/2309.02014>`_.

    Args:
      fun: Objective function. The function needs to have the optimization variable ``params`` as its first argument, and ``data`` argument to take data inputs.
      grad_fun: Optional gradient oracle that is expected to have the same function signature as the objective.
      hvp_fun: Optional hessian-vector-product oracle that takes optimization variable ``params`` as its first argument, vector ``v`` as its second arguments.
      pre_update: Optional gradient update function that gets called before every parameter update.
      rank: Rank of the preconditioner.
      rho: Regularization parameter (with non-negative value).
      precond_update_freq: A non-negative integer that specifies the update frequency of the
        preconditioner. When set to ``0`` or :math:`\infty` (*e.g.* ``jax.numpy.inf`` or
        ``numpy.inf``), the optimizer uses constant preconditioner that is constructed at
        the beginning of the optimization process.
      snapshop_update_freq: A non-negative integer that specifies how frequent snapshot points are computed.
      grad_batch_size:
      hess_batch_size:
      seed: An integer used as a seed to generate random numbers.
      learning_rate: Step size for applying updates (default :math:`0.5`). It can either be
        a fixed scalar value or a schedule based on step count. When set to ``None``, the
        algorithm adaptively chooses a learning rate whenever the preconditioner is updated.
      maxiter: Maximum number of iterations to run the solver.
      tol: Threshold of the gradient norm used for terminating the solver.
      verbose:

    """

    precond_update_freq: int
    snapshop_update_freq: int

    def _init_state(self) -> SketchySVRGState:
        return SketchySVRGState(
            iter_num=jnp.asarray(0),
            value=jnp.asarray(jnp.inf),
            error=jnp.asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=lambda g: g,
            lr=jnp.asarray(0.0),
        )

    def _update_params(
        self, params, state, params_snapshot, full_grad, data, *args, **kwargs
    ) -> SolverState:
        batch_idx, key = generate_random_batch(data, self.grad_batch_size, state.key)

        if self.pre_update:
            solver_state = self.pre_update(
                params, state, *args, **kwargs, data=data[batch_idx]
            )

        value, grad = self._value_and_grad(
            params, *args, **kwargs, data=data[batch_idx]
        )
        _, grad_snapshot = self._value_and_grad(
            params_snapshot, *args, **kwargs, data=data[batch_idx]
        )

        flattend_grad, unravel_fn = ravel_pytree(grad)
        flattend_grad_snapshot, _ = ravel_pytree(grad_snapshot)

        direction = unravel_fn(
            state.precond(flattend_grad - flattend_grad_snapshot + full_grad)
        )
        params = tree_add_scalar_mul(params, -state.lr, direction)
        error = tree_l2_norm(grad, squared=False)

        return SolverState(
            params=params,
            state=SketchySVRGState(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
                precond=state.precond,
                lr=state.lr,
            ),
        )

    def run(self, init_params: Any, data: ArrayLike, *args, **kwargs) -> SolverState:
        """Runs the optimization loop over an iterator.

        Args:
          init_params: pytree containing the initial parameters.
          data: iterator generating data batches.
          *args: additional positional arguments to be passed to ``fun`` (or ``grad_fun`` if provided).
          **kwargs: additional keyword arguments to be passed to ``fun`` (or ``grad_fun`` if provided).
        Returns:
          (params, state)
        """
        n = jnp.shape(data)[0]
        params_snapshot = init_params
        state = self._init_state()

        for i in range(self.maxiter):

            _, full_grad = self._value_and_grad(
                params_snapshot, *args, **kwargs, data=data
            )
            full_grad, _ = ravel_pytree(full_grad)
            params = params_snapshot

            for j in range(self.snapshop_update_freq):

                if (
                    self.precond_update_freq == 0
                    and i * self.snapshop_update_freq + j == 0
                ) or (
                    self.precond_update_freq > 0
                    and (i * self.snapshop_update_freq + j) % self.precond_update_freq
                    == 0
                ):
                    params, state = self._update_precond(
                        params, state, data, *args, **kwargs
                    )

                params, state = self._update_params(
                    params, state, params_snapshot, full_grad, data, *args, **kwargs
                )

            params_snapshot = params

            if state.error < self.tol:
                jax.debug.print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)  # type: ignore
