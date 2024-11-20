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
    tree_size,
)

from .promise import PromiseSolver

KeyArray = Array


class SketchySAGAState(NamedTuple):
    r"""The SketchySAGA optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      grad_table: Table of gradients of each individual component.
      table_avg: Average of the gradients in the table.
      step_size: Step size for the next update.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    grad_table: Array
    table_avg: Array
    step_size: Array


@dataclass(eq=False, kw_only=True)
class SketchySAGA(PromiseSolver):
    r"""The SketchySAGA optimizer.

    SketchySAGA is a preconditioned version of a minibatch variant [#f3]_ (b-nice SAGA)
    of SAGA [#f2]_. The optimizer maintains and updates a gradient table and table
    average. At each iteration, the optimizer computes auxiliary vector and uses it to
    update the variance-reduced stochastic gradient. The update is then based on the
    preconditioned variance-reduced stochastic gradient.

    .. note:: Because SketchySAGA computes gradient of each individual component
      function, the provided objective function ``fun`` or gradient function
      ``grad_fun`` needs to be compatible with 1-dimensional data input (*i.e.* when the
      data input is a vector representing a single sample). The following example
      expands the vector to a 2-dimensional array to handle the single sample case.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchySAGA

        def ridge_reg_objective(params, data, reg):
            # make data 2-dimensional if it is a vector of a single sample
            if jnp.ndim(data) == 1:
                data = jnp.expand_dims(data, axis=0)
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchySAGA(fun=ridge_reg_objective, ...)
        params, state = opt.run(init_params, data, reg)

    .. rubric:: References

    .. [#f2] A\. Defazio, F. Bach, and S. Lacoste-Julien, `SAGA: A Fast Incremental
      Gradient Method With Support for Non-Strongly Convex Composite Objectives
      <https://papers.nips.cc/paper_files/paper/2014/hash/
      ede7e2b6d13a41ddf9f4bdef84fdc737-Abstract.html>`_, Advances in Neural Information
      Processing Systems, 27, 2014.
    .. [#f3] N\. Gazagnadou, R. M. Gower, and J. Salmon, `Optimal Mini-Batch and Step
      Sizes for SAGA <https://proceedings.mlr.press/v97/gazagnadou19a.html>`_, in
      *Proceedings of the 36*\ :sup:`th` *International Conference on Machine Learning*,
      Proceedings of Machine Learning Research (PMLR), 97: 2142-2150, 2019.

    - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned
      Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates
      <https://arxiv.org/abs/2309.02014>`_.

    Args:
      fun: Scalar-valued objective function compatible with 1-dimensional ``data``
        input. :term:`↪ More Details<fun>`
      grad_fun: Optional gradient oracle corresponding to the provided objective
        function ``fun``. It also needs to be compatible with 1-dimensional ``data``
        input. :term:`↪ More Details<grad_fun>`
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

    learning_rate: Union[float, Callable] = 0.5

    def __post_init__(self):
        r"""The function overrides the superclass's method and constructs
        ``value_and_grad`` function that computes the gradient of each individual
        component."""
        if callable(self.grad_fun):
            comp_grad_fun = self.grad_fun
        else:
            comp_grad_fun = jax.grad(self.fun)

        def value_and_grad(params, data, reg, *args, **kwargs):
            g = lambda d: ravel_tree(
                comp_grad_fun(params, *args, **kwargs, data=d, reg=reg)
            )[0]
            grad_fun = jax.vmap(g, 0, 0)
            return self.fun(params, *args, **kwargs, data=data, reg=reg), grad_fun(data)

        self._value_and_grad = value_and_grad

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchySAGAState:
        r"""The function initializes the optimizer state."""
        return SketchySAGAState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            grad_table=inexact_asarray(jnp.zeros((self.num_samples, self.params_len))),
            table_avg=inexact_asarray(jnp.zeros((self.params_len,))),
            step_size=inexact_asarray(0.0),
        )

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.grad_batch_size,), replace=False
        )

        # compute stochastic gradients
        value, unraveled_grads = self._value_and_grad(
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )
        error = jnp.linalg.norm(jnp.mean(unraveled_grads, axis=0))

        # compute auxiliary
        aux = jnp.sum(unraveled_grads - state.grad_table[batch_idx], axis=0)

        # compute preconditioned variance-reduced stochastic gradient
        _, unravel_fun = ravel_tree(params)
        direction = unravel_fun(
            self._grad_transform(
                state.table_avg + (1.0 / self.grad_batch_size) * aux, state.precond
            ),
        )

        # update table average and gradient table
        table_avg = state.table_avg + (1.0 / self.num_samples) * aux
        grad_table = state.grad_table.at[batch_idx].set(unraveled_grads)

        # perform an update
        params = tree_add_scalar_mul(params, -state.step_size, direction)

        return SolverState(
            params=params,
            state=state._replace(
                iter_num=state.iter_num + 1,
                value=value,
                error=error,
                key=key,
                grad_table=grad_table,
                table_avg=table_avg,
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
          *args: Additional positional arguments to be passed to ``fun`` (and `
            `grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
        Returns:
          Final optimization variable and solver state. The variable has the same shape
          as the provided initial value ``init_params``, and the state is an
          :class:`SketchySAGAState` object.
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
