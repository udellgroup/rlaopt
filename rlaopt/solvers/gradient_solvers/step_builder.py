"""Helper module for building gradient solver step functions."""

from dataclasses import replace
from functools import partial
from typing import Callable

import torch

from rlaopt.data import DataLoader
from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import IdentityConfig
from rlaopt.solvers.gradient_solvers import gradient_solver_core as core
from rlaopt.splitting.operator_split import _OperatorSplit
from rlaopt.splitting.prox_grad_split import ProxGradSplit
from rlaopt.splitting.sapphire_split import SapphireSplit

from .gradient_solver_configs import GradSolverConfig, ProxGradConfig, SapphireConfig
from .gradient_solver_states import GradSolverState, SapphireState
from .precond_update_fn_builder import build_preconditioner_update

DataBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# Core step interface
def optimization_step(
    var_values: TensorDict,
    state: GradSolverState,
    var_values_transform: Callable,
    gradient_fn: Callable,
    apply_updates: Callable,
    update_precond_fn: Callable,
) -> tuple[TensorDict, GradSolverState]:
    """Execute one optimization step: precondition, gradient, update."""
    state = update_precond_fn(var_values, state)

    updates, state, var_values = _get_updates(
        var_values, state, var_values_transform, gradient_fn
    )

    var_values, state = apply_updates(updates, var_values, state)

    return var_values, state


def _get_updates(
    var_values: TensorDict,
    state: GradSolverState,
    var_values_transform: Callable,
    gradient_fn: Callable,
):
    var_values, state = var_values_transform(var_values, state)

    updates, state = gradient_fn(var_values, state)

    return updates, state, var_values


def get_step_fn(config: GradSolverConfig, op_split: ProxGradSplit | SapphireSplit):
    """Builds step function for gradient based solvers based on the config."""
    if isinstance(config, ProxGradConfig):
        return _prox_grad_step_builder(config, op_split)
    else:
        return _sapphire_step_builder(config, op_split)


##### ProxGrad Builder #####
def _prox_grad_step_builder(config: ProxGradConfig, op_split: _OperatorSplit):
    # Get functions used for building proximal gradient step
    f, grad_f, prox_fn = op_split.func_f, op_split.grad_f, op_split.prox

    err_fn = partial(core.grad_mapping_norm, full_gradient_fn=grad_f, prox_fn=prox_fn)

    if config.use_linesearch:
        # linesearch calls apply_updates(x, eta) -> TensorDict, so pass prox_fn directly
        apply_updates = partial(core.linesearch, f=f, apply_updates=prox_fn)
    else:
        # optimization_step calls apply_updates(updates, var_values, state); wrap to
        # match prox_update(var_vals, updates, state, prox_fn) argument order.
        def apply_updates(updates, var_vals, state):
            return core.prox_update(var_vals, updates, state, prox_fn)

    if config.use_acceleration:
        var_values_transform = core.nest_accel_update
    else:

        def var_values_transform(p, s):
            return p, s

    # grad_f(v) -> TensorDict; wrap to match (v, s) -> (updates, s) interface
    def gradient_fn(v, s):
        return grad_f(v), s

    def no_op_precond(v, s):
        return s

    _step = partial(
        optimization_step,
        var_values_transform=var_values_transform,
        gradient_fn=gradient_fn,
        apply_updates=apply_updates,
        update_precond_fn=no_op_precond,
    )

    def step(var_values, state):
        var_values, state = _step(var_values, state)
        var_values, state = err_fn(var_values, state)
        return var_values, replace(state, iter_=state.iter_ + 1)

    return step


##### SAPPHIRE builder #####
def _sapphire_step_builder(config: SapphireConfig, op_split: SapphireSplit):
    """Builds the step function for SAPPHIRE from the provided config."""
    n = op_split.num_samples
    grad_batch_size = op_split.loader.batch_size
    conv_factor = n // grad_batch_size
    # Create dataloader for the preconditioner and stepsize update
    precond_loader = DataLoader(
        op_split.loader.dataset, batch_size=op_split.loader.batch_size
    )

    # Get loss and prediction functions
    loss_fn, prediction_fn = op_split.model._loss_fn, op_split.model._get_prediction

    # Get batch and full gradient functions
    batch_grad_fn, full_gradient_fn = op_split.batch_grad_loss, op_split.grad_loss

    # Get prox operator and data loader
    prox_fn, loader = op_split.prox, op_split.loader

    # Get gradient oracle for base method specified in the config file.
    if config.base_method == "sgd":
        gradient_fn = core.SGDOracle.build_gradient_fn(
            loader=loader, batch_gradient_fn=batch_grad_fn
        )
    elif config.base_method == "svrg":
        # Get snapshot update frequency
        update_threshold = conv_factor * config.snapshot_update_freq
        gradient_fn = core.SVRGOracle.build_gradient_fn(
            loader=loader,
            batch_gradient_fn=batch_grad_fn,
            full_gradient_fn=full_gradient_fn,
            update_threshold=update_threshold,
        )
    else:
        gradient_fn = core.SAGAOracle.build_gradient_fn(
            loader=loader,
            loss_fn=loss_fn,
            prediction_fn=prediction_fn,
            grad_reg=op_split.grad_reg,
            n=n,
            has_intercept=op_split.model.fit_intercept,
        )

    # Build preconditioner update function
    device = op_split.model.variable_values.to_flat_tensor().device
    dtype = op_split.model.variable_values.to_flat_tensor().dtype

    # Convert from epochs to iterations
    check_termination_freq = config.check_termination_freq * conv_factor
    precond_update_freq = config.precond_update_freq * conv_factor

    update_precond_fn = build_preconditioner_update(
        config.precond_config,
        op_split,
        precond_loader,
        precond_update_freq,
        device,
        dtype,
        config.auto_update_stepsize,
    )

    # Branching condition
    # config_cond is True when an identity preconditioner is used
    # or the Nyström preconditioner is used and the objective is smooth.

    config_cond = isinstance(config.precond_config, IdentityConfig) or (
        not isinstance(config.precond_config, IdentityConfig) and op_split.r is None
    )

    if config_cond:

        def apply_updates(
            updates: TensorDict, var_values: TensorDict, state: SapphireState
        ):
            dir_ = core.precond_grad(updates, state)
            return core.prox_update(var_values, dir_, state, prox_fn)

    else:

        def apply_updates(
            updates: TensorDict, var_values: TensorDict, state: SapphireState
        ):
            return core.prox_update_P(
                var_values, updates, state, config.subproblem_iters, prox_fn
            )

    _step = partial(
        optimization_step,
        var_values_transform=lambda p, s: (p, s),
        gradient_fn=gradient_fn,
        apply_updates=apply_updates,
        update_precond_fn=update_precond_fn,
    )

    def step(
        var_values: TensorDict, state: SapphireState
    ) -> tuple[TensorDict, SapphireState]:
        var_values, state = _step(var_values, state)

        if state.iter_ % check_termination_freq == 0:
            var_values, state = core.grad_mapping_norm(
                var_values, state, full_gradient_fn, prox_fn
            )

        return var_values, replace(state, iter_=state.iter_ + 1)

    return step
