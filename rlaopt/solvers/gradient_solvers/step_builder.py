"""Helper module for building gradient solver step functions."""

from dataclasses import replace
from functools import partial
from typing import Callable

import torch

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

    prox_step = partial(
        core.prox_gd_step,
        full_gradient_fn=grad_f,
        prox_fn=prox_fn,
    )

    ls_step = partial(core.linesearch, f=f, full_gradient_fn=grad_f, prox_fn=prox_fn)

    # Only non-ls chains get err_fn as ls automatically
    # computes current error.
    err_fn = partial(core.grad_mapping_norm, full_gradient_fn=grad_f, prox_fn=prox_fn)

    #  Update chain for acceleration and linesearch
    if config.use_acceleration and config.use_linesearch:
        chain = (core.nest_accel_update, ls_step)

    # Update chain for just acceleration
    elif config.use_acceleration:
        chain = (core.nest_accel_update, prox_step, err_fn)

    # Update chain for just linesearch
    elif config.use_linesearch:
        chain = (ls_step,)

    # Update chain for vanilla proximal gradient.
    else:
        chain = (prox_step, err_fn)

    return _chain_updates(chain)


def _chain_updates(update_chain: tuple[Callable, ...]):
    def chained_updates(variable_values: TensorDict, state: GradSolverState):
        for update_fn in update_chain:
            variable_values, state = update_fn(variable_values, state)
        return variable_values, replace(state, iter_=state.iter_ + 1)

    return chained_updates


##### SAPPHIRE builder #####
def _sapphire_step_builder(config: SapphireConfig, op_split: SapphireSplit):
    """Builds the step function for SAPPHIRE from the provided config."""
    n = op_split.num_samples
    grad_batch_size = op_split.loader.batch_size
    conv_factor = n // grad_batch_size

    # Get loss and prediction functions
    loss_fn, prediction_fn = op_split.model._loss_fn, op_split.model._get_prediction

    # Get batch and full gradient functions
    batch_grad_fn, full_gradient_fn = op_split.batch_grad_loss, op_split.grad_loss

    # Get prox operator and data loader functions
    prox_fn, loader_fn = op_split.prox, op_split.loader.get_batch

    # Setup termination function
    termination_fn = partial(
        core.grad_mapping_norm, full_gradient_fn=full_gradient_fn, prox_fn=prox_fn
    )

    # Get gradient oracle for base method specified in the config file.
    if config.base_method == "sgd":
        gradient_fn = core.SGDOracle.build_gradient_fn(batch_grad_fn=batch_grad_fn)
    elif config.base_method == "svrg":
        # Get snapshot update frequency
        update_threshold = conv_factor * config.snapshot_update_freq
        gradient_fn = core.SVRGOracle.build_gradient_fn(
            batch_gradient_fn=batch_grad_fn,
            full_gradient_fn=full_gradient_fn,
            update_threshold=update_threshold,
        )
    else:
        gradient_fn = core.SAGAOracle.build_gradient_fn(
            loss_fn=loss_fn,
            prediction_fn=prediction_fn,
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
        precond_update_freq,
        device,
        dtype,
        config.auto_update_stepsize,
    )

    # Branching condition
    # config_cond is True when an identity preconditioner is used
    # or the Nyström preconditioner is used and there is no non-smooth term

    config_cond = isinstance(config.precond_config, IdentityConfig) or (
        not isinstance(config.precond_config, IdentityConfig) and op_split.r is None
    )

    if config_cond:
        return _sapphire_pipeline_chain(
            loader_fn,
            gradient_fn,
            core.precond_grad,
            prox_fn,
            update_precond_fn,
            termination_fn,
            check_termination_freq=check_termination_freq,
        )
    else:
        # Branch where non-identity preconditioner is used and there is a non-smooth
        # proxable regularizer present.
        prox_P_fn = partial(
            core.prox_update_P,
            subproblem_iters=config.subproblem_iters,
            prox_fn=prox_fn,
        )
    return _sapphire_pipeline_chain(
        loader_fn,
        gradient_fn,
        None,
        prox_P_fn,
        update_precond_fn,
        termination_fn,
        check_termination_freq=check_termination_freq,
    )


def _sapphire_pipeline_chain(
    loader_fn: Callable[[], DataBatch],
    gradient_fn: Callable[
        [TensorDict, SapphireState, DataBatch], tuple[TensorDict, SapphireState]
    ],
    transform_fn: Callable[[TensorDict, SapphireState], TensorDict] | None,
    update_fn: Callable[
        [TensorDict, TensorDict, SapphireState], tuple[TensorDict, SapphireState]
    ],
    update_precond_fn: Callable[[DataBatch, TensorDict, SapphireState], SapphireState],
    termination_fn: Callable,
    check_termination_freq: int = 100,
):
    """Pipeline: load data → preconditioner update → gradient → transform
    → variable update → check termination.
    """
    if transform_fn is None:
        transform_fn = core.identity_transform

    def step(variable_values: TensorDict, state: SapphireState):
        # Get batch
        batch = loader_fn()

        # Update preconditioner if needed
        state = update_precond_fn(variable_values, state, batch)

        # Compute gradient
        grads, state = gradient_fn(variable_values, state, batch)

        # Transform gradient (e.g., preconditioning)
        updates = transform_fn(grads, state)

        # Apply update
        variable_values, state = update_fn(variable_values, updates, state)

        if state.iter_ % check_termination_freq == 0:
            variable_values, state = termination_fn(variable_values, state)

        return variable_values, replace(state, iter_=state.iter_ + 1)

    return step
