from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse
from jax.typing import ArrayLike

from sketchyopts.base import (
    PromiseSolver,
    SolverState,
)

from sketchyopts.util import (
    inexact_asarray,
    integer_asarray,
    ravel_tree,
    tree_add,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_scalar_mul,
    tree_sub
)


KeyArray = Array
KeyArrayLike = ArrayLike

class SketchyKatyushaState(NamedTuple):
    r"""The SketchyKatyusha optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      full_grad: Full gradient at the snapshot.
      snapshot: Snapshot of the iterate.
      z: Momentum of the iterate.
      step_size: Step size for the next update.
      L: Smoothness constant estimate.
      sigma: Inverse condition number estimate.
      theta: Momentum parameter.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    full_grad: Array
    snapshot: Array
    z: Array
    step_size: Array
    L: Array
    sigma: Array
    theta: Array


@dataclass(eq=False, kw_only=True)
class SketchyKatyusha(PromiseSolver):
    r"""The SketchyKatyusha optimizer.

    SketchyKatyusha is a preconditioned version of Loopless Katyusha [#f5]_ that extends
    the original Katyusha [#f4]_. The optimizer calculates the preconditioned
    variance-reduced stochastic gradient and performs momentum update. The optimizer
    periodically updates the preconditioner, and probabilistically updates snapshot and
    full gradient.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchyKatyusha

        def ridge_reg_objective(params, data, reg):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchyKatyusha(fun=ridge_reg_objective, ...)
        params, state = opt.run(init_params, data, reg)

    .. rubric:: References

    .. [#f4] Z\. Allen-Zhu, `Katyusha: The First Direct Acceleration of Stochastic
      Gradient Methods <https://jmlr.org/papers/v18/16-410.html>`_, Journal of Machine
      Learning Research, 18(221): 1-51, 2018.
    .. [#f5] D\. Kovalev, S. Horváth, and P. Richtárik, `Don't Jump Through Hoops and
      Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop
      <https://proceedings.mlr.press/v117/kovalev20a.html>`_, in *Proceedings of the
      31*\ :sup:`th` *International Conference on Algorithmic Learning Theory*,
      Proceedings of Machine Learning Research (PMLR), 117: 451-467, 2020.

    - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned
      Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates
      <https://arxiv.org/abs/2309.02014>`_.

    Args:
      fun: Scalar-valued objective function. :term:`↪ More Details<fun>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. :term:`↪ More Details<grad_fun>`
      hvp_fun: Optional Hessian-vector product oracle for the Nyström subsampled Newton
        preconditioner. :term:`↪ More Details<hvp_fun>`
      sqrt_hess_fun: Required oracle that computes the square root of the Hessian matrix
        for the subsampled Newton preconditioner (*i.e.* cannot be empty if ``precond``
        is set to ``ssn``). :term:`↪ More Details<sqrt_hess_fun>`
      pre_update: Optional function to execute before optimizer's each update on the
        iterate. :term:`↪ More Details<pre_update>`
      precond: Type of preconditioner to use (default ``nyssn``). Either ``nyssn`` for
        Nyström subsampled Newton or ``ssn`` for subsampled Newton.
        :term:`↪ More Details<precond>`
      rho: Regularization parameter for the preconditioner. Expect a non-negative value
        (default ``1e-3``).
      rank: Rank of the Nyström subsampled Newton preconditioner. Expect a positive
        value (default ``10``).
      mu: Strong convexity parameter. Expect a positive value.
      grad_batch_size: Size of the batch of data to compute stochastic gradient at each
        iteration. Expect a positive value.
      hess_batch_size: Size of the batch of data to estimate the stochastic Hessian when
        updating the preconditioner. Expect a positive value.
      update_freq: Update frequency of the preconditioner. When set to ``0`` or
        :math:`\infty` (*e.g.* ``jax.numpy.inf`` or ``numpy.inf``), the optimizer uses
        constant preconditioner that is constructed at the beginning of the optimization
        process.
      snapshop_update_prob: Probability of updating the snapshot. Expect a value in
        :math:`(0,1)`.
      seed: Initial seed for the random number generator (default ``0``).
      momentum_param: Momentum parameter (default ``1/2``).
      momentum_multiplier: Momentum multiplier (default ``2/3``).
      maxiter: Maximum number of iterations to run the optimizer (default ``20``).
        Expect a positive value.
      tol: Threshold of the gradient norm used for terminating the optimizer (default
        ``1e-3``).
      verbose: Whether to print diagnostic message (default ``False``).
      jit: Whether to JIT-compile the optimization process (default ``True``).
        :term:`↪ More Details<jit>`
      sparse: Whether to sparsify the optimization process (default ``False``).
        :term:`↪ More Details<sparse>`
    """

    mu: float
    momentum_param: float = 1.0 / 2
    momentum_multiplier: float = 2.0 / 3
    snapshop_update_prob: float

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchyKatyushaState:
        r"""The function initializes the optimizer state."""
        _, full_grad = self._value_and_grad(params, *args, **kwargs, data=data, reg=reg)
        full_grad, _ = ravel_tree(full_grad)
        return SketchyKatyushaState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            full_grad=full_grad,
            snapshot=params,
            z=params,
            step_size=inexact_asarray(0.0),
            L=inexact_asarray(0.0),
            sigma=inexact_asarray(0.0),
            theta=inexact_asarray(0.0),
        )

    def _update_step_size(self, labda, state):
        r"""The function overrides the superclass's method and updates relevant values
        using the provided preconditioned smoothness constant labda."""
        sigma = self.mu / labda
        theta = jnp.minimum(
            0.5, jnp.sqrt(self.momentum_multiplier * sigma * self.num_samples)
        )
        step_size = self.momentum_param / (theta * (1 + self.momentum_param))
        return state._replace(
            step_size=step_size,
            L=labda,
            sigma=sigma,
            theta=theta,
        )

    def _update_snapshot(self, u, params, state, data, reg, *args, **kwargs):
        r"""The function updates snapshot and full gradient with specified update
        probability."""

        def update():
            _, full_grad = self._value_and_grad(
                params, *args, **kwargs, data=data, reg=reg
            )
            full_grad, _ = ravel_tree(full_grad)
            return state._replace(full_grad=full_grad, snapshot=params)

        return jax.lax.cond(u <= self.snapshop_update_prob, update, lambda: state)

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey1, subkey2 = jax.random.split(state.key, num=3)
        batch_idx = jax.random.choice(
            subkey1, self.num_samples, (self.grad_batch_size,), replace=False
        )
        u = jax.random.uniform(subkey2)

        # compute negative momentum
        x = tree_scalar_mul(state.theta, state.z)
        x = tree_add_scalar_mul(x, self.momentum_param, state.snapshot)
        x = tree_add_scalar_mul(x, 1 - state.theta - self.momentum_param, params)

        # compute preconditioned variance-reduced accelerated stochastic gradient
        value, grad = self._value_and_grad(
            x, *args, **kwargs, data=data[batch_idx], reg=reg
        )
        _, grad_snapshot = self._value_and_grad(
            state.snapshot,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )

        error = tree_l2_norm(grad, squared=False)

        unraveled_grad, unravel_fun = ravel_tree(grad)
        unraveled_grad_snapshot, _ = ravel_tree(grad_snapshot)

        direction = unravel_fun(
            self._grad_transform(
                unraveled_grad - unraveled_grad_snapshot + state.full_grad,
                state.precond,
            ),
        )

        # update snapshot
        state = self._update_snapshot(u, params, state, data, reg, *args, **kwargs)

        # perform an update
        z = tree_scalar_mul(state.step_size * state.sigma, x)
        z = tree_add(z, state.z)
        z = tree_add_scalar_mul(z, -state.step_size / state.L, direction)
        z = tree_scalar_mul(1.0 / (1 + state.step_size * state.sigma), z)
        params = tree_add_scalar_mul(x, state.theta, tree_sub(z, state.z))

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
                z=z,
            ),
        )

    def run(
        self, init_params: Any, data: ArrayLike, reg: float = 0.0, *args, **kwargs
    ) -> SolverState:
        r"""The function runs the optimization loop.

        Args:
          init_params: Initial value of the optimization variable.
          data: Full dataset. Expect an array of shape ``(num_samples, ...)``.
          reg: Regularization strength. Expect a non-negative value (default ``0``).
          *args: Additional positional arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`SketchyKatyushaState` object.
        """
        # initialize iterate and state
        params = init_params
        self.num_samples = jnp.shape(data)[0]
        self.params_len = jnp.size(ravel_tree(params)[0])
        state = self._init_state(params, data, reg, *args, **kwargs)

        # define partial functions
        update_params = lambda p, s: self._update_params(
            p, s, data, reg, *args, **kwargs
        )
        update_precond = lambda p, s: self._update_precond(
            p, s, data, reg, *args, **kwargs
        )

        # sparsify functions if needed
        if self.sparse:
            update_params = sparse.sparsify(update_params)
            update_precond = sparse.sparsify(update_precond)

        # JIT-compile functions if needed
        if self.jit:
            update_params = (
                jax.jit(update_params) if self.maxiter > 1 else update_params
            )
            update_precond = (
                jax.jit(update_precond)
                if (self.update_freq > 0 and self.maxiter / self.update_freq > 1)
                else update_precond
            )

        # run the optimization loop
        for i in range(self.maxiter):
            # call custom pre-update function
            if self.pre_update:
                params, state = self.pre_update(
                    params, state, *args, **kwargs, data=data, reg=reg
                )

            # update preconditioner
            if (self.update_freq == 0 and state.iter_num == 0) or (
                self.update_freq > 0 and state.iter_num % self.update_freq == 0
            ):
                state = update_precond(params, state)

            # update iterate
            params, state = update_params(params, state)

            # break out of loop if tolerance has been reached
            if state.error < self.tol:
                print(
                    "Info: early termination because error tolerance has been reached."
                )
                break

        return SolverState(params=params, state=state)

