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
    method ``_update_precond`` that updates the Nyström subsampled Newton (NySSN) or subsampled Newton (SSN)
    preconditioner and the corresponding preconditioned smoothness constant. Lastly, it generates the function
    that applies preconditioner to gradient.

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates <https://arxiv.org/abs/2309.02014>`_.

    """

    fun: Callable
    grad_fun: Optional[Callable] = None
    hvp_fun: Optional[Callable] = None
    sqrt_hess_fun: Optional[Callable] = None
    pre_update: Optional[Callable] = None

    precond: str = "nyssn"

    rho: float
    rank: int = 10
    grad_batch_size: int
    hess_batch_size: int
    update_freq: int
    seed: int

    maxiter: int = 20
    tol: float = 1e-3
    verbose: bool = False
    jit: bool = True

    def __post_init__(self):
        r"""The function constructs ``_value_and_grad`` that evaluates both the objective and gradient."""
        # validate preconditioner type and required oracle
        if self.precond == "ssn":
            if not self.sqrt_hess_fun:
                raise ValueError(
                    "Unspecified square root Hessian oracle for the subsampled Newton preconditioner."
                )
        elif self.precond != "nyssn":
            raise ValueError(
                'Invalid preconditioner type: argument precond must be specified to either "nyssn" for Nyström subsampled Newton or "ssn" for subsampled Newton.'
            )

        # initialize values to appease static type checker
        self.num_samples = 0
        self.params_len = 0
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

    def _get_grad_transform_nyssn(self, U, S):
        r"""The function constructs ``grad_transform`` that applies preconditioner to gradient for the Nyström subsampled Newton preconditioner."""

        @jax.jit
        def grad_transform(g):
            UTg = U.T @ g
            return U @ (UTg / (S + self.rho)) + g / self.rho - U @ UTg / self.rho

        return grad_transform

    def _get_grad_transform_ssn(self, L, X):
        r"""The function constructs ``grad_transform`` that applies preconditioner to gradient for the subsampled Newton preconditioner."""

        if jnp.shape(X)[0] < self.params_len:

            @jax.jit
            def grad_transform(g):
                v = X @ g
                v = jnp.linalg.solve(L, v)
                v = jnp.linalg.solve(L.T, v)
                v = X.T @ v
                return (1 / self.rho) * (g - v)

            return grad_transform

        @jax.jit
        def grad_transform(g):
            v = jnp.linalg.solve(L, g)
            v = jnp.linalg.solve(L.T, v)
            return v

        return grad_transform

    def _estimate_constant(
        self, H_S, grad_transform, key, p_tol: float = 1e-5, p_maxiter: int = 20
    ):
        r"""The function estimates the preconditioned smoothness constant."""

        # stopping criterion
        def cond_fun(value):
            _, _, norm_r, k = value
            return (norm_r >= p_tol) & (k < p_maxiter)

        # power iteration
        def body_fun(value):
            y, _, _, k = value
            y_ = grad_transform(y)
            y_ = H_S @ y_
            labda = jnp.dot(y, y_)
            norm_r = jnp.linalg.norm(y_ - labda * y)
            y = y_ / jnp.linalg.norm(y_)
            return y, labda, norm_r, k + 1

        # initialization
        y = jax.random.normal(key, (self.params_len,))
        y = y / jnp.linalg.norm(y)
        initial_value = (y, 0, p_tol, 0)

        _, labda, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_value)

        return labda

    def _update_step_size(self, labda, state):
        r"""The function updates the step size using the learning rate from the class variable and the provided smoothness constant."""
        return state._replace(step_size=self.learning_rate / labda)

    def _update_precond(self, params, state, data, reg, *args, **kwargs):
        r"""The function updates the Nyström subsampled Newton (NySSN) or subsampled Newton (SSN) preconditioner and corresponding smoothness constant."""
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.hess_batch_size,), replace=False
        )
        H_S = HessianLinearOperator(
            fun=self.fun,
            grad_fun=self.grad_fun,
            hvp_fun=self.hvp_fun,
            sqrt_hess_fun=self.sqrt_hess_fun,
            params=params,
            data=data[batch_idx],
            reg=0,
            *args,
            **kwargs,
        )

        # update NySSN preconditioner
        if self.precond == "nyssn":
            key, subkey = jax.random.split(key)
            U, S = sketchyopts.preconditioner.rand_nystrom_approx(
                H_S, self.rank, subkey
            )
            grad_transform = self._get_grad_transform_nyssn(U, S)
        # update SSN preconditioner
        else:
            X = self.sqrt_hess_fun(params, *args, **kwargs, data=data[batch_idx])  # type: ignore
            if jnp.shape(X)[0] >= self.params_len:
                H = X.T @ X
            else:
                H = X @ X.T
            L = jnp.linalg.cholesky(
                H + self.rho * jnp.identity(jnp.shape(H)[0]), upper=False
            )
            grad_transform = self._get_grad_transform_ssn(L, X)

        # update step size rate with explicitly specified learning schedule
        if callable(self.learning_rate):
            step_size = self.learning_rate(state.iter_num)
            state = state._replace(step_size=step_size)
        # update step size with smoothness constant estimate
        else:
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.choice(
                subkey, self.num_samples, (self.hess_batch_size,), replace=False
            )
            H_S = HessianLinearOperator(
                fun=self.fun,
                grad_fun=self.grad_fun,
                hvp_fun=self.hvp_fun,
                sqrt_hess_fun=self.sqrt_hess_fun,
                params=params,
                data=data[batch_idx],
                reg=reg,
                *args,
                **kwargs,
            )
            labda = self._estimate_constant(H_S, grad_transform, subkey)
            state = self._update_step_size(labda, state)

        return state._replace(
            key=key,
            precond=jax.tree_util.Partial(grad_transform),
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

    def __init__(
        self,
        fun,
        params,
        data,
        reg,
        grad_fun=None,
        hvp_fun=None,
        sqrt_hess_fun=None,
        *args,
        **kwargs,
    ):
        r"""Initialize the Hessian linear operator.

        The linear operator implicitly forms the Hessian of function ``fun`` with respect
        to parameters ``params``. The function can have additional positional and keyword
        arguments.

        The operator uses automatic differentiation to compute Hessian-vector product,
        unless an oracle ``hvp_fun`` is provided.

        The operator uses automatic differentiation to construct the full Hessian matrix,
        unless ``sqrt_hess_fun`` is provided.

        Args:
          fun: Scalar-valued function.
          grad_fun: Optional gradient oracle.
          hvp_fun: Optional Hessian-vector product oracle.
          sqrt_hess_fun: Optional oracle that computes the matrix form (2-dimensional array) of the square root of the Hessian (with respect to the objective that does not includes the regularization term).
          params: Parameters of the function.
          data: Data to feed into the function.
          reg: Regularization strength.
          *args: Additional positional arguments to be passed to ``fun`` (and ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and ``grad_fun``, ``hvp_fun``, ``sqrt_hess_fun`` if provided).

        """
        unraveled, unravel_fun = sketchyopts.util.ravel_tree(params)
        params_size = jnp.size(unraveled)

        # construct Hessian-vector product function
        if hvp_fun:
            self.hvp_fn = lambda v: sketchyopts.util.ravel_tree(
                hvp_fun(params, unravel_fun(v), *args, **kwargs, data=data, reg=reg)
            )[0]
        else:
            if grad_fun:
                # grad_fun_partial = lambda x: grad_fun(unravel_fun(x), *args, **kwargs, data=data, reg=reg)
                # hvp_fn = lambda v: jax.jvp(grad_fun_partial, [unraveled], [v])[1]
                # self.hvp_fn = lambda v: sketchyopts.util.ravel_tree(hvp_fn(v))[0]
                grad_fun_partial = lambda x: sketchyopts.util.ravel_tree(
                    grad_fun(unravel_fun(x), *args, **kwargs, data=data, reg=reg)
                )[0]
                self.hvp_fn = lambda v: jax.jvp(grad_fun_partial, [unraveled], [v])[1]
            else:
                fun_partial = lambda x: fun(
                    unravel_fun(x), *args, **kwargs, data=data, reg=reg
                )
                self.hvp_fn = lambda v: jax.jvp(
                    jax.grad(fun_partial), [unraveled], [v]
                )[1]

        # construct Hessian matrix function
        if sqrt_hess_fun:

            def h_mat_fn():
                H = sqrt_hess_fun(params, *args, **kwargs, data=data)
                return H.T @ H + reg * jnp.identity(params_size)

            self.h_mat_fn = h_mat_fn
        else:
            if grad_fun:
                grad_fun_partial = lambda x: sketchyopts.util.ravel_tree(
                    grad_fun(unravel_fun(x), *args, **kwargs, data=data, reg=reg)
                )[0]
                self.h_mat_fn = lambda: jax.jacfwd(grad_fun_partial)(unraveled)
            else:
                fun_partial = lambda x: fun(
                    unravel_fun(x), *args, **kwargs, data=data, reg=reg
                )
                self.h_mat_fn = lambda: jax.hessian(fun_partial)(unraveled)

        super().__init__(shape=(params_size, params_size), ndim=2)

    def matmul(self, other):
        r"""Compute the Hessian-vector or Hessian-matrix product.

        The vector or matrix ``other`` the Hessian acts on must have the matching size of the parameters ``params``.
        Specifically, the function expects ``(params_size)`` for a vector or ``(params_size, num_vectors)`` for a matrix.
        If ``params`` is a pytree, the size should be the number of leaves of the tree (*i.e.* length of the flattened tree).

        The resulting array has size ``(params_size)`` for vector input or ``(params_size, num_vectors)`` for matrix input.

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

    def as_matrix(self):
        r"""Construct the full Hessian matrix.

        The function materializes the full Hessian matrix of size ``(params_size, params_size)``. Here ``params_size``
        refers to the size of the (flattened) parameters ``params``. For instance, if ``params`` is a pytree, the size
        is the number of leaves of the tree (*i.e.* length of the flattened tree).

        Returns:
          A 2-dimensional array representing the Hessian matrix.

        """
        return self.h_mat_fn()
