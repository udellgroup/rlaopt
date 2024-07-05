import abc
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import sketchyopts

KeyArray = Array
KeyArrayLike = ArrayLike


class SolverState(NamedTuple):
    r"""Class for encapsulating parameters and solver state.

    Args:
      params: Parameters the solver seeks to optimize.
      state: State of the solver.
    """

    params: Any
    state: Any


@dataclass(eq=False, kw_only=True)
class PromiseSolver(abc.ABC):
    r"""Base class for PROMISE solvers.

    The class provides basic functionalities of the sketching-based preconditioned stochastic gradient algorithms.

    It constructs function ``_value_and_grad`` that evaluates both the objective and gradient. It also implements
    method ``_update_precond`` that updates the Nystroö subsampled Newton (NySSN) preconditioner and the corresponding
    preconditioned smoothness constant. Lastly, it generates the function that applies preconditioner to gradient.

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates <https://arxiv.org/abs/2309.02014>`_.

    """

    fun: Callable
    grad_fun: Optional[Callable] = None
    hvp_fun: Optional[Callable] = None
    pre_update: Optional[Callable] = None

    rho: float
    rank: int
    grad_batch_size: int
    hess_batch_size: int
    update_freq: int
    seed: int

    maxiter: int = 20
    tol: float = 1e-3
    verbose: bool = False

    def __post_init__(self):
        r"""The function constructs ``_value_and_grad`` that evaluates both the objective and gradient."""
        # initialize values to appease static type checker
        self.num_samples = 0
        if "learning_rate" not in dir(self):
            self.learning_rate = None

        # construct objective value and gradient function
        if self.grad_fun is None:
            self._value_and_grad = jax.value_and_grad(self.fun)
        else:

            def value_and_grad(params, *args, **kwargs):
                return self.fun(params, *args, **kwargs), self.grad_fun(  # type: ignore
                    params, *args, **kwargs
                )

            self._value_and_grad = value_and_grad

    def _get_grad_transform(self, U, S):
        r"""The function constructs ``grad_transform`` that applies preconditioner to gradient."""

        @jax.jit
        def grad_transform(g):
            UTg = U.T @ g
            return U @ (UTg / (S + self.rho)) + g / self.rho - U @ UTg / self.rho

        return grad_transform

    def _get_inv_sqrt_precond(self, U, S):
        r"""The function constructs ``matvec`` that computes matrix-vector product for the inverse square-root of preconditioner"""

        @jax.jit
        def matvec(v):
            UTv = U.T @ v
            return U @ (UTv / jnp.sqrt(S + self.rho)) + (1 / self.rho**0.5) * (
                v - U @ UTv
            )

        return matvec

    def _estimate_constant(
        self, H_S, inv_sqrt_precond, key, p_tol: float = 1e-5, p_maxiter: int = 20
    ):
        r"""The function estimates the preconditioned smoothness constant."""
        n = jnp.shape(H_S)[0]

        # stopping criterion
        def cond_fun(value):
            _, _, norm_r, k = value
            return (norm_r >= p_tol) & (k < p_maxiter)

        # power iteration
        def body_fun(value):
            y, _, _, k = value
            y_ = inv_sqrt_precond(y)
            y_ = H_S @ y_
            y_ = inv_sqrt_precond(y_)
            labda = jnp.dot(y, y_)
            norm_r = jnp.linalg.norm(y_ - labda * y)
            y = y_ / jnp.linalg.norm(y_)
            return y, labda, norm_r, k + 1

        # initialization
        y = jax.random.normal(key, (n,))
        y = y / jnp.linalg.norm(y)
        initial_value = (y, 0, p_tol, 0)

        _, labda, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_value)

        return labda  # type: ignore

    def _update_step_size(self, labda, state):
        r"""The function updates the step size using the learning rate from the class variable and the provided smoothness constant."""
        return state._replace(step_size=self.learning_rate / labda)

    def _update_precond(self, params, state, data, *args, **kwargs):
        r"""The function updates the Nyström subsampled Newton (NySSN) preconditioner and corresponding smoothness constant."""
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.hess_batch_size,), replace=False
        )
        H_S = HessianLinearOperator(
            self.fun,
            self.grad_fun,
            self.hvp_fun,
            params,
            *args,
            **kwargs,
            data=data[batch_idx],
        )
        U, S = sketchyopts.preconditioner.rand_nystrom_approx(H_S, self.rank, subkey)

        # explicitly specified learning schedule
        if callable(self.learning_rate):
            step_size = self.learning_rate(state.iter_num)
            state = state._replace(step_size=step_size)
        # adaptive learning rate
        else:
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.choice(
                subkey, self.num_samples, (self.hess_batch_size,), replace=False
            )
            H_S = HessianLinearOperator(
                self.fun,
                self.grad_fun,
                self.hvp_fun,
                params,
                *args,
                **kwargs,
                data=data[batch_idx],
            )
            inv_sqrt_precond = self._get_inv_sqrt_precond(U, S)
            labda = self._estimate_constant(H_S, inv_sqrt_precond, subkey)
            state = self._update_step_size(labda, state)

        return state._replace(
            key=key,
            precond=jax.tree_util.Partial(self._get_grad_transform(U, S)),
        )


class LinearOperator(abc.ABC):
    r"""Base interface for abstract linear operators."""

    def __init__(self, shape: tuple, ndim: int):
        r"""Initialize the linear operator.

        Args:
          shape: Shape of the linear operator.
          ndim: Dimension of the linear operator.

        """
        self.shape = shape
        self.ndim = ndim

    @abc.abstractmethod
    def matmul(self, other: ArrayLike) -> Array:
        r"""Compute a matrix-vector or matrix-matrix product between the operator and
        a JAX array.

        Args:
          other: JAX array with matching dimension.

        Returns:
          A JAX array representing the resulting vector or matrix.

        """

    def __matmul__(self, other: ArrayLike) -> Array:
        r"""An alias for function :func:`sketchyopts.base.LinearOperator.matmul`.

        This overwrites the ``@`` operator.

        """
        return self.matmul(other)


class HessianLinearOperator(LinearOperator):
    r"""Hessian operator for computing Hessian-vector product without explicitly forming
    the Hessian matrix.
    """

    def __init__(self, fun, grad_fun, hvp_fun, params, *args, **kwargs):
        r"""Initialize the Hessian linear operator.

        The linear operator implicitly forms the Hessian of function ``fun`` with respect
        to parameters ``params``. The function can have additional positional and keyword
        arguments.

        The operator uses automatic differentiation to compute Hessian-vector product,
        unless an oracle ``hvp_fun`` is provided.

        Args:
          fun: Scalar-valued function.
          grad_fun: Optional gradient oracle.
          hvp_fun: Optional Hessian-vector product oracle.
          params: Parameters of the function.
          *args: Additional positional arguments to be passed to ``fun`` (and ``grad_fun`` as well as ``hvp_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and ``grad_fun`` as well as ``hvp_fun`` if provided).

        """
        unraveled, unravel_fun = sketchyopts.util.ravel_tree(params)
        params_size = jnp.size(unraveled)

        if hvp_fun:
            self.hvp_fn = lambda v: sketchyopts.util.ravel_tree(
                hvp_fun(params, unravel_fun(v), *args, **kwargs)
            )[0]
        else:
            if grad_fun:
                grad_fun_partial = lambda x: grad_fun(x, *args, **kwargs)
                hvp = lambda v: jax.jvp(grad_fun_partial, [params], [unravel_fun(v)])[1]
            else:
                fun_partial = lambda x: fun(x, *args, **kwargs)
                hvp = lambda v: jax.jvp(
                    jax.grad(fun_partial), [params], [unravel_fun(v)]
                )[1]
            self.hvp_fn = lambda v: sketchyopts.util.ravel_tree(hvp(v))[0]

        super().__init__(shape=(params_size, params_size), ndim=2)

    def matmul(self, other):
        r"""Compute the Hessian-vector or Hessian-matrix product.

        The vector or matrix ``other`` the Hessian acts on must have the matching size of the parameters ``params``.
        Specifically, the function expects ``(params_size)`` for a vector or ``(params_size, num_vectors)`` for a matrix.
        If ``params`` is a pytree, the size should be the number of leaves of the tree (*i.e.* length of the flattened tree).

        The resulting array has ``(params_size)`` for vector input or ``(params_size, num_vectors)`` for matrix input.

        Args:
          other: A 1-dimensional or 2-dimensional array with the matching size.

        Returns:
          Array representing the result.

        """
        if jnp.ndim(other) == 1:
            return self.hvp_fn(other)
        elif jnp.ndim(other) == 2:
            return jax.vmap(self.hvp_fn, 1, 1)(other)
        else:
            raise sketchyopts.errors.InputDimError("operand 1", jnp.ndim(other), [1, 2])
