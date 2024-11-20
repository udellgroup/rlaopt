from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse
from jax.typing import ArrayLike

from sketchyopts.base import SolverState
from sketchyopts.util import (
    inexact_asarray,
    integer_asarray,
    ravel_tree,
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_size,
)

from .promise import PromiseSolver

KeyArray = Array


class SketchySVRGState(NamedTuple):
    r"""The SketchySVRG optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      full_grad: Full gradient at the snapshot.
      snapshot: Snapshot of the iterate.
      step_size: Step size for the next update.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    full_grad: Array
    snapshot: Array
    step_size: Array


@dataclass(eq=False, kw_only=True)
class SketchySVRG(PromiseSolver):
    r"""The SketchySVRG optimizer.

    SketchySVRG is a preconditioned version of SVRG [#f1]_ (stochastic variance reduced
    gradient). The optimizer stores and periodically updates a snapshot of the iterate,
    and uses the snapshot and its full gradient to perform variance reduction. The
    preconditioning is then applied to the variance-reduced stochastic gradient at each
    step. The preconditioner and learning rate get updated periodically.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchySVRG

        def ridge_reg_objective(params, reg, data):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchySVRG(fun=ridge_reg_objective, ...)
        params, state = opt.run(init_params, data, reg)

    .. rubric:: References

    .. [#f1] R\. Johnson and T. Zhang, `Accelerating Stochastic Gradient Descent using
      Predictive Variance Reduction <https://papers.nips.cc/paper/4937-accelerating-
      stochastic-gradient-descent-using-predictive-variance-reduction>`_, Advances in
      Neural Information Processing Systems, 26, 2013.

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
      grad_batch_size: Size of the batch of data to compute stochastic gradient at each
        iteration. Expect a positive value.
      hess_batch_size: Size of the batch of data to estimate the stochastic Hessian when
        updating the preconditioner. Expect a positive value.
      update_freq: Update frequency of the preconditioner. When set to ``0`` or
        :math:`\infty` (*e.g.* ``jax.numpy.inf`` or ``numpy.inf``), the optimizer uses
        constant preconditioner that is constructed at the beginning of the optimization
        process.
      snapshop_update_freq: Update frequency of the snapshot. Expect a positive value.
      seed: Initial seed for the random number generator (default ``0``).
      learning_rate: Step size for applying updates (default ``0.5``). It can either be
        a fixed scalar value or a schedule (callable) based on step count.
        :term:`↪ More Details<learning_rate>`
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

    snapshop_update_freq: int
    learning_rate: Union[float, Callable] = 0.5

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchySVRGState:
        r"""The function initializes the optimizer state."""
        return SketchySVRGState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            full_grad=inexact_asarray(jnp.zeros((self.params_len,))),
            snapshot=params,
            step_size=inexact_asarray(0.0),
        )

    def _update_snapshot(
        self, params, state, data, reg, *args, **kwargs
    ) -> SketchySVRGState:
        _, full_grad = self._value_and_grad(params, *args, **kwargs, data=data, reg=reg)
        unraveled_full_grad, _ = ravel_tree(full_grad)
        return state._replace(
            full_grad=unraveled_full_grad,
            snapshot=params,
        )

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.grad_batch_size,), replace=False
        )

        # compute preconditioned variance-reduced stochastic gradient
        value, grad = self._value_and_grad(
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
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

        # perform an update
        params = tree_add_scalar_mul(params, -state.step_size, direction)

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
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
          :class:`SketchySVRGState` object.
        """
        # initialize iterate and state
        params = init_params
        self.num_samples = jnp.shape(data)[0]
        self.params_len = tree_size(params)
        state = self._init_state(params, data, reg, *args, **kwargs)

        # define partial functions
        update_params = lambda p, s: self._update_params(
            p, s, data, reg, *args, **kwargs
        )
        update_precond = lambda p, s: self._update_precond(
            p, s, data, reg, *args, **kwargs
        )
        update_snapshot = lambda p, s: self._update_snapshot(
            p, s, data, reg, *args, **kwargs
        )

        # sparsify functions if needed
        if self.sparse:
            update_params = sparse.sparsify(update_params)
            update_precond = sparse.sparsify(update_precond)
            update_snapshot = sparse.sparsify(update_snapshot)

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
            update_snapshot = (
                jax.jit(update_snapshot)
                if self.maxiter / self.snapshop_update_freq > 1
                else update_snapshot
            )

        # run the optimization loop
        for i in range(self.maxiter):
            # call custom pre-update function
            if self.pre_update:
                params, state = self.pre_update(
                    params, state, *args, **kwargs, data=data, reg=reg
                )

            # update snapshot
            if state.iter_num % self.snapshop_update_freq == 0:
                state = update_snapshot(params, state)

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
