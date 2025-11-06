"""Proximal gradient solver implementation."""

from dataclasses import dataclass, replace
from typing import Callable

import torch
from pydantic import Field

from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.operator_split import OperatorSplit
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria
from rlaopt.solvers.solver_base import OptimSolver, SolverState
from rlaopt.solvers.utils import split_objective


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


class ProxGradStoppingCriteria(StoppingCriteria):
    """Stopping criteria specific to the Proximal Gradient solver.

    Attributes:
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence based on the error metric.
    """

    tol: float = Field(default=1e-4, gt=0)


@dataclass(frozen=True)
class ProxGradState(SolverState):
    """State container for the proximal gradient solver.

    Attributes:
        iter_: Current iteration count.
        eta: Step size (learning rate) for the gradient descent step.
        variable_values_prev: Previous iteration's variable values,
            used for accelerated methods. None when acceleration is disabled.
        err: Current error metric, measuring convergence progress.
            Initialized to infinity.
    """

    eta: float = 1.0
    variable_values_prev: TensorDict | None = None
    err: torch.Tensor = torch.tensor(torch.inf)


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
        self, obj: Expression | AddExpression | OperatorSplit, config: ProxGradConfig
    ):
        """Initialize the proximal gradient solver.

        Args:
            obj: The optimization objective.
            config: Configuration for the solver.
        """
        super().__init__(obj, config)
        self._init_state = _build_init_state(config.eta, config.use_acceleration)
        self._step = _build_step(obj, config.use_acceleration, config.use_linesearch)
        self._solve = lambda tol, max_iters: _build_solve(
            obj, self._init_state, self._step, tol, max_iters
        )

    def init_state(self, variable_values: TensorDict) -> ProxGradState:
        """Initialize the solver state.

        Args:
            variable_values: Initial variable values.

        Returns:
            Initial solver state.
        """
        return self._init_state(variable_values)

    def step(
        self, variable_values: TensorDict, state: ProxGradState
    ) -> tuple[TensorDict, ProxGradState]:
        """Perform a single optimization step.

        Args:
            variable_values: Current variable values.
            state: Current solver state.

        Returns:
            Tuple of updated variable values and state.
        """
        return self._step(variable_values, state)

    def solve(
        self,
        variable_values: TensorDict | None = None,
        stopping_criteria: ProxGradStoppingCriteria = ProxGradStoppingCriteria(),
    ) -> tuple[TensorDict, torch.Tensor]:
        """Solve the optimization problem using the proximal gradient method.

        Args:
            variable_values: Initial variable values. If None, defaults to
                variable values in objective.
            stopping_criteria: Criteria to determine when to stop the optimization.
                Defaults to ProxGradStoppingCriteria().

        Returns:
            Tuple of optimized variable values and final solver error.
        """
        solve_fn = self._solve(
            tol=stopping_criteria.tol, max_iters=stopping_criteria.max_iters
        )
        return solve_fn(variable_values)


def _build_init_state(
    eta: float, use_acceleration: bool
) -> Callable[[TensorDict], ProxGradState]:
    """Build the function to initialize the solver state."""

    def init_state(variable_values: TensorDict) -> ProxGradState:
        """Initialize the solver state."""
        state_inputs = {"iter_": 0, "eta": eta}

        if use_acceleration:
            state_inputs["variable_values_prev"] = variable_values

        return ProxGradState(**state_inputs)

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
    def err_fn(var_vals: TensorDict, state: ProxGradState) -> torch.Tensor:
        grads = obj.grad_f(var_vals)
        updated_var_vals = var_vals - state.eta * grads
        prox_var_vals = obj.prox(updated_var_vals, state.eta)

        return (var_vals - prox_var_vals).flat_norm() / state.eta

    if use_acceleration and use_linesearch:

        def step(
            var_vals: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            var_vals_prev = var_vals
            var_vals, state = _accel_prox_grad_ls_step(var_vals, state, f, grad_f, prox)
            return var_vals, replace(
                state, iter_=state.iter_ + 1, var_vals_prev=var_vals_prev
            )

    elif use_acceleration:

        def step(
            var_vals: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            var_vals_prev = var_vals
            var_vals = _accel_prox_grad_step(var_vals, state, grad_f, prox)
            err = err_fn(var_vals, state)
            return var_vals, replace(
                state, iter_=state.iter_ + 1, err=err, var_vals_prev=var_vals_prev
            )

    elif use_linesearch:

        def step(
            var_vals: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            var_vals, state = _linesearch(var_vals, state, f, grad_f, prox)
            return var_vals, replace(state, iter_=state.iter_ + 1)

    else:

        def step(
            var_vals: TensorDict, state: ProxGradState
        ) -> tuple[TensorDict, ProxGradState]:
            var_vals = _prox_grad_step(var_vals, state, grad_f, prox)
            err = err_fn(var_vals, state)
            return var_vals, replace(state, iter_=state.iter_ + 1, err=err)

    return step


def _build_solve(
    obj: Expression | AddExpression | OperatorSplit,
    init_state_fn: Callable,
    step_fn: Callable,
    tol: float,
    max_iters: int,
) -> Callable[[TensorDict | None], tuple[TensorDict, torch.Tensor]]:
    """Build the solve function with stopping criteria."""

    def solve(var_vals: TensorDict | None = None) -> tuple[TensorDict, torch.Tensor]:
        """Solve the optimization problem."""
        if var_vals is None:
            if isinstance(obj, OperatorSplit):
                var_vals = obj.f.variable_values
            else:
                var_vals = obj.variable_values

        state = init_state_fn(var_vals)

        # Get error tolerance
        epsilon = tol * (var_vals.flat_dim()) ** 0.5

        while state.err > epsilon and state.iter_ < max_iters:
            var_vals, state = step_fn(var_vals, state)

        return var_vals, state.err

    return solve


def _prox_grad_step(
    var_vals: TensorDict,
    state: ProxGradState,
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    """Perform a basic proximal gradient step."""
    grads = grad_f(var_vals)
    var_vals = _prox_update(var_vals, grads, state, prox)
    return var_vals


def _accel_prox_grad_ls_step(
    var_vals: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> tuple[TensorDict, ProxGradState]:
    """Perform an accelerated proximal gradient step with line search."""
    y = _accel_step(var_vals, state)
    var_vals, state = _linesearch(y, state, f, grad_f, prox)
    return var_vals, state


def _accel_prox_grad_step(
    var_vals: TensorDict,
    state: ProxGradState,
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    """Perform an accelerated proximal gradient step."""
    y = _accel_step(var_vals, state)
    var_vals = _prox_grad_step(y, state, grad_f, prox)
    return var_vals


def _accel_step(var_vals: TensorDict, state: ProxGradState) -> TensorDict:
    """Compute the accelerated (momentum) step."""
    momentum_scale = state.iter_ / (state.iter_ + 3)
    return var_vals + momentum_scale * (var_vals - state.var_vals_prev)


def _linesearch(
    var_vals: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> tuple[TensorDict, ProxGradState]:
    """Perform backtracking line search to find appropriate step size."""
    beta = 0.5
    f0 = f(var_vals)
    grads = grad_f(var_vals)
    cond = False

    def linesearch_step(var_vals: TensorDict, state: ProxGradState):
        z = _prox_update(var_vals, grads, state, prox)
        d = z - var_vals
        u = f0 + grads.flat_dot(d) + 1 / (2 * state.eta) * (d.flat_norm() ** 2)

        if f(z) <= u:
            err_new = d.flat_norm() / state.eta
            return True, z, replace(state, err=err_new)
        else:
            eta_new = beta * state.eta
            return False, var_vals, replace(state, eta=eta_new)

    while not cond:
        cond, var_vals, state = linesearch_step(var_vals, state)

    return var_vals, state


def _prox_update(
    var_vals: TensorDict,
    grads: TensorDict,
    state: ProxGradState,
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    """Apply the proximal update: prox(var_vals - eta * grads, eta)."""
    return prox(var_vals - state.eta * grads, state.eta)
