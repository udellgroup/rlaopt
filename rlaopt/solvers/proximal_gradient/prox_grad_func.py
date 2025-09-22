from math import sqrt
from typing import Callable, NamedTuple, Optional, Tuple

import torch

from rlaopt.solvers.configs import ProxGradConfig
from rlaopt.solvers.utils import split_objective
from rlaopt.expression.expression import Expression
from rlaopt.operator_split import OperatorSplit
from rlaopt.utils import tensor_dict_ops as dict_ops
from rlaopt._typing import TensorDict


class ProxGradState(NamedTuple):
    eta: torch.Tensor
    params_prev: Optional[TensorDict] = None
    err: torch.Tensor = torch.inf
    iter_: int = 0


def proximal_gradient(
    obj: Expression | OperatorSplit,
    config: ProxGradConfig,
    init_params: Optional[TensorDict] = None,
) -> Tuple[TensorDict, torch.Tensor]:

    # Unpack config
    eta, max_iters, tol, use_acceleration, use_linesearch = (
        config.eta,
        config.max_iters,
        config.tol,
        config.use_acceleration,
        config.use_linesearch,
    )

    # If no stepsize, set initial stepsize:
    if not eta:
        eta = 1.0
        # Use linesearch if it is not enabled
        if not use_linesearch:
            use_linesearch = True

    # Build step function and initialize optimizer params and state
    step = _build_step(obj, use_acceleration, use_linesearch)

    # Initialize params and optimizer state
    if init_params:
        params = init_params

    elif isinstance(obj, OperatorSplit):
        params = obj.f.params
    
    else:
        params = obj.params

    state = _init_state(params, eta, use_acceleration)

    # Get error tolerance
    epsilon = tol * sqrt(dict_ops.dim(params))

    # Solver loop
    while state.err > epsilon and state.iter_ <= max_iters:
        params, state = step(params, state)
    return params, state.err


def _init_state(
    params: TensorDict, eta: torch.Tensor, use_acceleration: bool
) -> ProxGradState:
    if use_acceleration:
        return ProxGradState(params_prev=params, eta=eta)
    else:
        return ProxGradState(eta=eta)


def _build_step(
    obj: Expression | OperatorSplit, use_acceleration: bool, use_linesearch: bool
) -> Callable[
    [TensorDict, ProxGradState],
    Tuple[TensorDict, ProxGradState],
]:

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
        ) -> Tuple[TensorDict, ProxGradState]:
            params_prev = params
            params, state = _accel_prox_grad_ls_step(
                params, state, f, grad_f, prox
            )
            return params, state._replace(
                iter_=state.iter_ + 1, params_prev=params_prev
            )

    elif use_acceleration:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> Tuple[TensorDict, ProxGradState]:
            params_prev = params
            params = _accel_prox_grad_step(params, state, grad_f, prox)
            err = err_fn(params, state)
            return params, state._replace(
                iter_=state.iter_ + 1, err=err, params_prev=params_prev
            )

    elif use_linesearch:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> Tuple[TensorDict, ProxGradState]:
            params, state = _linesearch(params, state, f, grad_f, prox)
            return params, state._replace(iter_=state.iter_ + 1)

    else:

        def step(
            params: TensorDict, state: ProxGradState
        ) -> Tuple[TensorDict, ProxGradState]:
            params = _prox_grad_step(params, state, grad_f, prox)
            err = err_fn(params, state)
            return params, state._replace(iter_=state.iter_ + 1, err=err)

    return step

def _prox_grad_step(
    params: TensorDict,
    state: ProxGradState,
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    grads = grad_f(params)
    params = _prox_update(params, grads, state, prox)
    return params


def _accel_prox_grad_ls_step(
    params: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> Tuple[TensorDict, ProxGradState]:
    y = _accel_step(params, state)
    params, state = _linesearch(y, state, f, grad_f, prox)
    return params, state


def _accel_prox_grad_step(
    params: TensorDict,
    state: ProxGradState,
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> TensorDict:
    y = _accel_step(params, state)
    params = _prox_grad_step(y, state, grad_f, prox)
    return params


def _accel_step(
    params: TensorDict, state: ProxGradState
) -> TensorDict:
    momentum_scale = state.iter_ / (state.iter_ + 3)
    momentum = dict_ops.scal_mul(dict_ops.sub(params, state.params_prev), momentum_scale)
    return dict_ops.add(params, momentum)

def _linesearch(
    params: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    grad_f: Callable[[TensorDict], TensorDict],
    prox: Callable[[TensorDict, float], TensorDict],
) -> Tuple[TensorDict, ProxGradState]:
    beta = 0.5
    f0 = f(params)
    grads = grad_f(params)
    cond = False

    def linesearch_step(params: TensorDict, state: ProxGradState):
        z = _prox_update(params, grads, state, prox)
        d = dict_ops.sub(z, params)
        u = f0 + dict_ops.dot(grads, d) + 1 / (2 * state.eta) * (dict_ops.elem_norm(d) ** 2)
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
    # prox(params - state.eta * grads, state.eta)
    return prox(
        dict_ops.sub(params, dict_ops.scal_mul(grads, state.eta)), 
        state.eta
    )
