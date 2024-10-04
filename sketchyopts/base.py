import abc
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import sketchyopts
from sketchyopts.util import inexact_asarray

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

    The class provides basic functionalities of the sketching-based preconditioned
    stochastic gradient algorithms.

    It constructs function ``_value_and_grad`` that evaluates both the objective and
    gradient. It also implements
    method ``_init_precond`` that initializes empty preconditioner decomposition, and
    ``_update_precond`` that updates the Nyström subsampled Newton (NySSN) or subsampled
    Newton (SSN) preconditioner as well as the corresponding preconditioned smoothness
    constant. Helper method ``_grad_transform`` applies preconditioner to the provided
    gradient vector.

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `PROMISE: Preconditioned
        Stochastic Optimization Methods by Incorporating Scalable Curvature Estimates
        <https://arxiv.org/abs/2309.02014>`_.
    """

    fun: Callable
    grad_fun: Optional[Callable] = None
    hvp_fun: Optional[Callable] = None
    sqrt_hess_fun: Optional[Callable] = None
    pre_update: Optional[Callable] = None

    precond: str = "nyssn"

    rho: float = 1e-3
    rank: int = 10
    grad_batch_size: int
    hess_batch_size: int
    update_freq: int
    seed: int = 0

    maxiter: int = 20
    tol: float = 1e-3
    verbose: bool = False
    jit: bool = True
    sparse: bool = False

    def __post_init__(self):
        r"""The function constructs ``_value_and_grad`` that evaluates both the
        objective and gradient."""
        # validate preconditioner type and required oracle
        if self.precond == "ssn":
            if not callable(self.sqrt_hess_fun):
                raise ValueError(
                    "Invalid square-root Hessian oracle for the subsampled Newton preconditioner."
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
        if not callable(self.grad_fun):
            self._value_and_grad = jax.value_and_grad(self.fun)
        else:

            def value_and_grad(params, *args, **kwargs):
                return self.fun(params, *args, **kwargs), self.grad_fun(
                    params, *args, **kwargs
                )  # type: ignore

            self._value_and_grad = value_and_grad

    def _init_precond(self, params, data, reg, *args, **kwargs):
        r"""The function initializes zero preconditioner of the correct shape."""
        # NySSN preconditioner
        if self.precond == "nyssn":
            precond = (
                sketchyopts.util.inexact_asarray(
                    jnp.zeros((self.params_len, self.rank))
                ),
                sketchyopts.util.inexact_asarray(jnp.zeros((self.rank,))),
            )
        # SSN preconditioner
        else:
            # get the shape of the square root of the Hessian
            X = self.sqrt_hess_fun(
                params, *args, **kwargs, data=data[: self.hess_batch_size]
            )  # type: ignore
            sqrt_hess_size = jnp.shape(X)[0]

            if sqrt_hess_size < self.params_len:
                precond = (
                    sketchyopts.util.inexact_asarray(
                        jnp.zeros((sqrt_hess_size, sqrt_hess_size))
                    ),
                    sketchyopts.util.inexact_asarray(
                        jnp.zeros((sqrt_hess_size, self.params_len))
                    ),
                )
            else:
                precond = (
                    sketchyopts.util.inexact_asarray(
                        jnp.zeros((self.params_len, self.params_len))
                    ),
                    sketchyopts.util.inexact_asarray(
                        jnp.zeros((sqrt_hess_size, self.params_len))
                    ),
                )

        return precond

    def _grad_transform(self, g, precond):
        r"""The function performs preconditioning on the gradient."""
        # NySSN preconditioner
        if self.precond == "nyssn":
            U, S = precond
            UTg = U.T @ g
            return U @ (UTg / (S + self.rho)) + g / self.rho - U @ UTg / self.rho
        # SSN preconditioner
        else:
            L, X = precond

            if jnp.shape(X)[0] < self.params_len:
                v = X @ g
                v = jnp.linalg.solve(L, v)
                v = jnp.linalg.solve(L.T, v)
                v = X.T @ v
                return (1.0 / self.rho) * (g - v)
            else:
                v = jnp.linalg.solve(L, g)
                v = jnp.linalg.solve(L.T, v)
                return v

    def _estimate_constant(
        self, H_S, precond, key, p_tol: float = 1e-5, p_maxiter: int = 20
    ):
        r"""The function estimates the preconditioned smoothness constant."""

        # stopping criterion
        def cond_fun(value):
            _, _, norm_r, k = value
            return (norm_r >= p_tol) & (k < p_maxiter)

        # power iteration
        def body_fun(value):
            y, _, _, k = value
            y_ = self._grad_transform(y, precond)
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
        r"""The function updates the step size using the learning rate from the class
        variable and the provided smoothness constant."""
        return state._replace(step_size=self.learning_rate / labda)

    def _update_precond(self, params, state, data, reg, *args, **kwargs):
        r"""The function updates the Nyström subsampled Newton (NySSN) or subsampled
        Newton (SSN) preconditioner and corresponding smoothness constant."""
        key, subkey = jax.random.split(state.key)
        batch_idx = jax.random.choice(
            subkey, self.num_samples, (self.hess_batch_size,), replace=False
        )

        # update NySSN preconditioner
        if self.precond == "nyssn":
            H_S = HessianLinearOperator(
                fun=self.fun,
                grad_fun=self.grad_fun,
                hvp_fun=self.hvp_fun,
                params=params,
                *args,
                **kwargs,
                data=data[batch_idx],
                reg=0,
            )
            key, subkey = jax.random.split(key)
            U, S = sketchyopts.preconditioner.rand_nystrom_approx(
                H_S, self.rank, subkey
            )
            precond = (U, S)
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
            precond = (L, X)

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
                params=params,
                *args,
                **kwargs,
                data=data[batch_idx],
                reg=reg,
            )
            labda = self._estimate_constant(H_S, precond, subkey)
            state = self._update_step_size(labda, state)
        return state._replace(
            key=key,
            precond=precond,
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
        r"""Compute a matrix-vector or matrix-matrix product between the operator and a
        JAX array.

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
    the Hessian matrix."""

    def __init__(
        self,
        fun,
        params,
        grad_fun=None,
        hvp_fun=None,
        *args,
        **kwargs,
    ):
        r"""Initialize the Hessian linear operator.

        The linear operator implicitly forms the Hessian of function ``fun`` with
        respect to parameters ``params``. The function can have additional positional
        and keyword arguments.

        The operator uses automatic differentiation to compute Hessian-vector product,
        unless an oracle ``hvp_fun`` is provided.

        Args:
          fun: Scalar-valued function.
          grad_fun: Optional gradient oracle.
          hvp_fun: Optional Hessian-vector product oracle.
          params: Parameters of the function.
          *args: Additional positional arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun`` if provided).
          **kwargs: Additional keyword arguments to be passed to ``fun`` (and
            ``grad_fun``, ``hvp_fun`` if provided).
        """
        unraveled, unravel_fun = sketchyopts.util.ravel_tree(params)
        params_size = jnp.size(unraveled)

        # construct Hessian-vector product function
        if hvp_fun:
            self.hvp_fn = lambda v: sketchyopts.util.ravel_tree(
                hvp_fun(params, unravel_fun(v), *args, **kwargs)
            )[0]
        else:
            if grad_fun:
                grad_fun_partial = lambda x: sketchyopts.util.ravel_tree(
                    grad_fun(unravel_fun(x), *args, **kwargs)
                )[0]
                self.hvp_fn = lambda v: jax.jvp(grad_fun_partial, [unraveled], [v])[1]
            else:
                fun_partial = lambda x: fun(unravel_fun(x), *args, **kwargs)
                self.hvp_fn = lambda v: jax.jvp(
                    jax.grad(fun_partial), [unraveled], [v]
                )[1]

        super().__init__(shape=(params_size, params_size), ndim=2)

    def matmul(self, other):
        r"""Compute the Hessian-vector or Hessian-matrix product.

        The vector or matrix ``other`` the Hessian acts on must have the matching size
        of the parameters ``params``. Specifically, the function expects
        ``(params_size)`` for a vector or ``(params_size, num_vectors)`` for a matrix.
        If ``params`` is a pytree, the size should be the number of leaves of the tree
        (*i.e.* length of the flattened tree).

        The resulting array has size ``(params_size)`` for vector input or
        ``(params_size, num_vectors)`` for matrix input.

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


class AddLinearOperator(LinearOperator):
    r"""Linear operator for adding two other linear operators together."""

    def __init__(self, operator1, operator2):
        r"""Construct the linear operator.

        The function forms a new linear operator :math:`\mathrm{operator1}
        + \mathrm{operator2}`.

        Args:
          operator1: First linear operator.
          operator2: Second linear operator.
        """
        if operator1.shape != operator1.shape:
            raise ValueError("Incompatible linear operator shapes.")
        self.operator1 = operator1
        self.operator2 = operator2
        super().__init__(shape=jnp.shape(self.operator1), ndim=jnp.ndim(self.operator1))

    def matmul(self, other):
        r"""Compute the matrix-vector or matrix-matrix product of the new operator."""
        return self.operator1 @ other + self.operator2 @ other
