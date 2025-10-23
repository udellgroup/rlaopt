"""Proximal gradient solver implementation."""

from typing import Callable

import torch
from pydantic import Field

from rlaopt._typing import OptimState, TensorDict
from rlaopt.expression.expression import AddExpression, Expression
from rlaopt.operator_split import OperatorSplit
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria
from rlaopt.solvers.solver_base import OptimSolver
from rlaopt.solvers.utils import split_objective
from rlaopt.utils import tensor_dict_ops as dict_ops


class ProxGradConfig(SolverConfig):
    """Configuration for the proximal gradient solver.

    Attributes:
        eta: Step size for the gradient update.
        use_acceleration: Whether to use acceleration techniques.
        use_linesearch: Whether to use line search for step size selection.
    """

    eta: float = Field(default=1.0, gt=0)
    use_acceleration: bool = False
    use_linesearch: bool = True


class ProxGradState(OptimState):
    """State container for the proximal gradient solver.

    Attributes:
        eta: Step size (learning rate) for the gradient descent step.
        params_prev: Previous iteration's parameters, used for acceleration methods.
            None when acceleration is disabled.
        err: Current error metric, measuring convergence progress.
            Initialized to infinity.
        iter_: Current iteration count, starting from 0.
    """

    eta: float
    params_prev: TensorDict | None = None
    err: torch.Tensor = torch.inf
    iter_: int = 0


class ProxGradStoppingCriteria(StoppingCriteria):
    """Stopping criteria specific to the Proximal Gradient solver.

    Attributes:
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence based on the error metric.
    """

    tol: float = Field(default=1e-4, gt=0)


class ProxGrad(OptimSolver):
    """Proximal gradient solver for optimization problems.

    Solves problems of the form:
        minimize f(x) + g(x)
    where f is smooth (differentiable) and g is proxable (has an efficient
    proximal operator).

    Supports multiple variants:
    - Basic proximal gradient (fixed step size)
    - Accelerated proximal gradient (Nesterov momentum)
    - Backtracking line search for adaptive step sizes
    - Combinations of acceleration and line search
    """

    def __init__(
        self, config: ProxGradConfig, obj: Expression | AddExpression | OperatorSplit
    ):
        """Initialize the proximal gradient solver.

        Args:
            config: Configuration for the solver.
            obj: The optimization objective.
        """
        super().__init__(config, obj)
        self._init_state = _build_init_state(config.eta, config.use_acceleration)
        self._step = _build_step(obj, config.use_acceleration, config.use_linesearch)
        self._solve = lambda tol, max_iters: _build_solve(
            obj, self._init_state, self._step, tol, max_iters
        )

    def init_state(self, params: TensorDict) -> ProxGradState:
        """Initialize the solver state.

        Args:
            params: Initial parameters.

        Returns:
            Initial solver state.
        """
        return self._init_state(params)

    def step(
        self, params: TensorDict, state: ProxGradState
    ) -> tuple[TensorDict, ProxGradState]:
        """Perform a single optimization step.

        Args:
            params: Current parameters.
            state: Current solver state.

        Returns:
            Tuple of updated parameters and state.
        """
        return self._step(params, state)

    def solve(
        self,
        stopping_criteria: ProxGradStoppingCriteria,
        params: TensorDict | None = None,
    ) -> tuple[TensorDict, torch.Tensor]:
        """Solve the optimization problem using the proximal gradient method.

        Args:
            stopping_criteria: Criteria to determine when to stop the optimization.
            params: Initial parameters. If None, defaults to zeros.

        Returns:
            Tuple of optimized parameters and final solver state.
        """
        solve_fn = self._solve(
            tol=stopping_criteria.tol, max_iters=stopping_criteria.max_iters
        )
        return solve_fn(params)


def _build_init_state(
    eta: float, use_acceleration: bool
) -> Callable[[TensorDict], ProxGradState]:
    """Build the function to initialize the solver state."""

    def init_state(params: TensorDict) -> ProxGradState:
        """Initialize the solver state."""
        if use_acceleration:
            return ProxGradState(params_prev=params, eta=eta)
        else:
            return ProxGradState(eta=eta)

    return init_state


def _build_step(
    obj: Expression | AddExpression | OperatorSplit,
    use_acceleration: bool,
    use_linesearch: bool,
) -> Callable[
    [TensorDict, ProxGradState],
    tuple[TensorDict, ProxGradState],
]:
    """Build the step function based on configuration."""
    # If the objective is not already an OperatorSplit, split it
    if isinstance(obj, Expression):
        obj = split_objective(obj)

    # Extract the function, gradient, and prox operator
    f, grad_f, prox = obj.f_func, obj.grad_f, obj.prox

    # Setup function computing stopping criteria
    def err_fn(params: TensorDict, state: ProxGradState) -> torch.Tensor:
        grads = obj.grad_f(params)
        updated_params = dict_ops.sub(params, dict_ops.scal_mul(grads, state.eta))
        prox_params = obj.prox(updated_params, state.eta)
        delta_params = dict_ops.sub(params, prox_params)
        err = dict_ops.elem_norm(delta_params) / state.eta
        return err

    if use_acceleration and use_linesearch:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            params_prev = params
            params, state = _accel_prox_grad_ls_step(params, state, f, grad_f, prox)
            return params, state._replace(
                iter_=state.iter_ + 1, params_prev=params_prev
            )

    elif use_acceleration:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            params_prev = params
            params = _accel_prox_grad_step(params, state, grad_f, prox)
            err = err_fn(params, state)
            return params, state._replace(
                iter_=state.iter_ + 1, err=err, params_prev=params_prev
            )

    elif use_linesearch:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            params, state = _linesearch(params, state, f, grad_f, prox)
            return params, state._replace(iter_=state.iter_ + 1)

    else:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            params = _prox_grad_step(params, state, grad_f, prox)
            err = err_fn(params, state)
            return params, state._replace(iter_=state.iter_ + 1, err=err)

    return step


def _build_solve(
    obj: Expression | AddExpression | OperatorSplit,
    init_state_fn: Callable,
    step_fn: Callable,
    tol: float,
    max_iters: int,
) -> Callable[[TensorDict | None], tuple[TensorDict, torch.Tensor]]:
    """Build the solve function with stopping criteria."""

    def solve(params: TensorDict | None = None) -> tuple[TensorDict, torch.Tensor]:
        """Solve the optimization problem."""
        if params is None:
            if isinstance(obj, OperatorSplit):
                params = obj.f.params
            else:
                params = obj.params

        state = init_state_fn(params)

        # Get error tolerance
        epsilon = tol * dict_ops.dim(params) ** 0.5

        while state.err > epsilon and state.iter_ < max_iters:
            params, state = step_fn(params, state)

        return params, state.err

    return solve


def _prox_grad_step(
    params: TensorDict,
    state: ProxGradState,
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    """Perform a basic proximal gradient step."""
    grads = grad_f(params)
    params = _prox_update(params, grads, state, prox)
    return params


def _accel_prox_grad_ls_step(
    params: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> tuple[TensorDict, ProxGradState]:
    """Perform an accelerated proximal gradient step with line search."""
    y = _accel_step(params, state)
    params, state = _linesearch(y, state, f, grad_f, prox)
    return params, state


def _accel_prox_grad_step(
    params: TensorDict,
    state: ProxGradState,
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    """Perform an accelerated proximal gradient step."""
    y = _accel_step(params, state)
    params = _prox_grad_step(y, state, grad_f, prox)
    return params


def _accel_step(params: TensorDict, state: ProxGradState) -> TensorDict:
    """Compute the accelerated (momentum) step."""
    momentum_scale = state.iter_ / (state.iter_ + 3)
    momentum = dict_ops.scal_mul(
        dict_ops.sub(params, state.params_prev), momentum_scale
    )
    return dict_ops.add(params, momentum)


def _linesearch(
    params: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> tuple[TensorDict, ProxGradState]:
    """Perform backtracking line search to find appropriate step size."""
    beta = 0.5
    f0 = f(params)
    grads = grad_f(params)
    cond = False

    def linesearch_step(params: TensorDict, state: ProxGradState):
        z = _prox_update(params, grads, state, prox)
        d = dict_ops.sub(z, params)
        u = (
            f0
            + dict_ops.dot(grads, d)
            + 1 / (2 * state.eta) * (dict_ops.elem_norm(d) ** 2)
        )
        if f(z) <= u:
            err_new = dict_ops.elem_norm(d) / state.eta
            return True, z, state._replace(err=err_new)
        else:
            eta_new = beta * state.eta
            return False, params, state._replace(eta=eta_new)

    while not cond:
        cond, params, state = linesearch_step(params, state)

    return params, state


def _prox_update(
    params: TensorDict,
    grads: TensorDict,
    state: ProxGradState,
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    """Apply the proximal update: prox(params - eta * grads, eta)."""
    return prox(dict_ops.sub(params, dict_ops.scal_mul(grads, state.eta)), state.eta)
