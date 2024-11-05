from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Union

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
    tree_add_scalar_mul,
    tree_l2_norm
)

KeyArray = Array
KeyArrayLike = ArrayLike


class SketchySGDState(NamedTuple):
    r"""The SketchySGD optimizer state.

    Args:
      iter_num: Number of iterations the optimizer has performed.
      value: Objective value at the current iterate.
      error: Gradient norm of the current iterate.
      key: PRNG key for the next update.
      precond: Decomposition of the preconditioner for the next update.
      step_size: Step size for the next update.
    """

    iter_num: Array
    value: Array
    error: Array
    key: KeyArray
    precond: tuple[Array, Array]
    step_size: Array


@dataclass(eq=False, kw_only=True)
class SketchySGD(PromiseSolver):
    r"""The SketchySGD optimizer.

    SketchySGD is a stochastic quasi-Newton method that uses sketching to approximate
    the curvature of the loss function. It maintains a preconditioner for SGD
    (stochastic gradient descent) using subsampled Hessian and automatically selects an
    appropriate learning whenever it updates the preconditioner.

    Example:
      .. highlight:: python
      .. code-block:: python

        import jax.numpy as jnp
        from sketchyopts.solver import SketchySGD

        def ridge_reg_objective(params, reg, data):
            # data has dimension num_samples * (feature_dim + 1)
            X, y = data[:,:feature_dim], data[:,feature_dim:]
            residuals = jnp.dot(X, params) - y
            return jnp.mean(residuals ** 2) + 0.5 * reg * jnp.dot(w ** 2)

        opt = SketchySGD(fun=ridge_reg_objective, ...)
        params, state = opt.run(init_params, data, reg)

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `SketchySGD: Reliable
        Stochastic Optimization via Randomized Curvature Estimates
        <https://arxiv.org/abs/2211.08597>`_.

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
      precond: Type of preconditioner to use. Either ``nyssn`` for Nyström subsampled
        Newton or ``ssn`` for subsampled Newton (default ``nyssn``).
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

    def _init_state(self, params, data, reg, *args, **kwargs) -> SketchySGDState:
        r"""The function initializes the optimizer state."""
        return SketchySGDState(
            iter_num=integer_asarray(0),
            value=inexact_asarray(jnp.inf),
            error=inexact_asarray(jnp.inf),
            key=jax.random.PRNGKey(self.seed),
            precond=self._init_precond(params, data, reg, *args, **kwargs),
            step_size=inexact_asarray(0.0),
        )

    def _update_params(self, params, state, data, reg, *args, **kwargs) -> SolverState:
        r"""The function performs an update on the iterate."""
        # generate a random batch
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.grad_batch_size,), replace=False
        )

        # compute preconditioned stochastic gradient
        value, grad = self._value_and_grad(
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
            reg=reg,
        )
        error = tree_l2_norm(grad, squared=False)
        unraveled_grad, unravel_fun = ravel_tree(grad)
        direction = unravel_fun(self._grad_transform(unraveled_grad, state.precond))

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
          :class:`SketchySGDState` object.
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