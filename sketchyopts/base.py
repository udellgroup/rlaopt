import jax
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree

import sketchyopts

import abc
from dataclasses import dataclass
from typing import NamedTuple, Any, Callable, Tuple, Union, Optional
from jax.typing import ArrayLike
from jax import Array

KeyArray = Array
KeyArrayLike = ArrayLike


class SolverState(NamedTuple):
    params: Any
    state: Any


@dataclass(eq=False, kw_only=True)
class PromiseSolver(abc.ABC):

    fun: Callable
    grad_fun: Optional[Callable] = None
    hvp_fun: Optional[Callable] = None
    pre_update: Optional[Callable] = None

    rank: int
    rho: float
    grad_batch_size: int
    hess_batch_size: int
    seed: int
    learning_rate: Union[float, Callable] = 0.5

    maxiter: int = 20
    tol: float = 1e-3
    verbose: bool = False

    def __post_init__(self):
        if self.grad_fun is None:
            self._value_and_grad = jax.value_and_grad(self.fun)
        else:

            def value_and_grad(params, *args, **kwargs):
                return self.fun(params, *args, **kwargs), self.grad_fun(  # type: ignore
                    params, *args, **kwargs
                )

            self._value_and_grad = value_and_grad

    def _get_grad_transform(self, U, S):

        @jax.jit
        def grad_transform(g):
            UTg = U.T @ g
            return U @ (UTg / (S + self.rho)) + g / self.rho - U @ UTg / self.rho

        return grad_transform

    # matrix-vector product for the inverse square-root of preconditioner
    def _get_inv_sqrt_precond(self, U, S):

        @jax.jit
        def matvec(v):
            UTv = U.T @ v
            return U @ (UTv / jnp.sqrt(S + self.rho)) + (1 / self.rho**0.5) * (
                v - U @ UTv
            )

        return matvec

    def _get_lr(
        self, H_S, inv_sqrt_precond, key, p_tol: float = 1e-5, p_maxiter: int = 10
    ):

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

        return self.learning_rate / labda  # type: ignore

    def _update_precond(self, params, state, data, *args, **kwargs) -> SolverState:

        key, subkey = jax.random.split(state.key)
        batch_idx, key = sketchyopts.generate_random_batch(
            data, self.hess_batch_size, key
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
        U, S = sketchyopts.rand_nystrom_approx(H_S, self.rank, subkey)

        # explicitly specified learning schedule
        if callable(self.learning_rate):
            lr = self.learning_rate(state.iter_num)
        # adaptive learning rate
        else:
            key, subkey = jax.random.split(key)
            batch_idx, key = sketchyopts.generate_random_batch(
                data, self.hess_batch_size, key
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
            lr = self._get_lr(H_S, inv_sqrt_precond, subkey)

        new_state = state._replace(
            key=key, precond=self._get_grad_transform(U, S), lr=lr
        )

        return SolverState(params=params, state=new_state)


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
        r"""Compute a matrix-vector or matrix-matrix product between this operator and
        a JAX array.

        Args:
          other: JAX array with matching dimension.

        Returns:
          A JAX array representing the resulting vector or matrix.

        """

    def __matmul__(self, other: ArrayLike) -> Array:
        r"""An alias for function :func:`sketchyopts.util.LinearOperator.matmul`.

        This overwrites the ``@`` operator.

        """
        return self.matmul(other)


class HessianLinearOperator(LinearOperator):
    r"""Hessian operator for computing hessian-vector-product without explicitly forming
    the Hessian matrix.
    """

    def __init__(self, fun, grad_fun, hvp_fun, params, *args, **kwargs):
        r"""Initialize the Hessian linear operator.

        The linear operator implcitly forms the Hessian of function ``f`` with respect
        to parameters ``params``. The function can have other arguments ``f_extra_args``
        that go into its evaluation, but only the second partial derivative
        information of parameters ``params`` gets used.

        Args:
          f: Scalar-valued function.
          params: Parameters of the function. Can be of any PyTree structure.
          **f_extra_args: Other optional arguments to the function.

        """
        unraveled, unravel_fn = ravel_pytree(params)
        params_size = jnp.size(unraveled)

        if hvp_fun:
            self.hvp_fn = lambda v: hvp_fun(params, v, *args, **kwargs)
        else:
            if grad_fun:
                grad_fun_partial = lambda x: grad_fun(x, *args, **kwargs)
                hvp = lambda v: jax.jvp(grad_fun_partial, [params], [unravel_fn(v)])[1]
            else:
                fun_partial = lambda x: fun(x, *args, **kwargs)
                hvp = lambda v: jax.jvp(
                    jax.grad(fun_partial), [params], [unravel_fn(v)]
                )[1]
            self.hvp_fn = lambda v: ravel_pytree(hvp(v))[0]

        super().__init__(shape=(params_size, params_size), ndim=2)

    def matmul(self, other):
        r"""Compute the Hessian-vector or Hessian-matrix product.

        The vector of matrix the Hessian acts on must be a JAX Array (not a PyTree).
        The result is also a JAX Array of the same dimension.

        Args:
          other: A 1D or 2D JAX array with matching size to the Hessian.

        Returns:
          JAX Array representing the result.

        """
        if jnp.ndim(other) == 1:
            return self.hvp_fn(other)
        elif jnp.ndim(other) == 2:
            return jax.vmap(self.hvp_fn, 1, 1)(other)
        else:
            raise ValueError(
                "matmul input operand 1 must have ndim 1 or 2, but it has ndim {}".format(
                    jnp.ndim(other)
                )
            )
