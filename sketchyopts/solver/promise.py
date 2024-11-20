import abc
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import sketchyopts
from sketchyopts.operator import HessianLinearOperator

KeyArray = Array
KeyArrayLike = ArrayLike


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
            U, S = sketchyopts.nystrom.rand_nystrom_approx(H_S, self.rank, subkey)
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
