from dataclasses import replace
from functools import partial
from typing import Callable

from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import IdentityConfig, NystromConfig, PreconditionerConfig
from rlaopt.linalg.preconditioners.nystrom import Nystrom
from rlaopt.splitting.operator_split import _OperatorSplit
from rlaopt.splitting.prox_grad_split import ProxGradSplit
from rlaopt.splitting.sapphire_split import SapphireSplit

import torch

from rlaopt.solvers.gradient_solvers import functional_core as core
from .optim_configs import GradSolverConfig, ProxGradConfig, SapphireConfig
from .optim_states import GradSolverState, SapphireState

def get_step_fn(config: GradSolverConfig, op_split: ProxGradSplit | SapphireSplit):
     if isinstance(config, ProxGradConfig):
          return _prox_grad_step_builder(config, op_split)
     else:
          return _sapphire_step_builder(config, op_split)

##### ProxGrad Builder #####
def _prox_grad_step_builder(config: ProxGradConfig, op_split: _OperatorSplit):
     prox_step = _op_split_partial(core.prox_gd_step, op_split)
     ls_update_par = _op_split_partial(core.linesearch, op_split)
     err_fn = _op_split_partial(core.grad_mapping_norm, op_split)

     if config.use_acceleration and config.use_linesearch:
               chain = (
                    core.nest_accel_update,
                    ls_update_par,
                    err_fn
               )

     elif config.use_acceleration:
               chain = (
                    core.nest_accel_update,
                    prox_step,
                    err_fn
               )
     elif config.use_linesearch:
               chain = (
                    ls_update_par,
                    err_fn
               )
     else:
          chain = (
               prox_step,
               err_fn
          )
     
     return _chain_updates(chain)

def _chain_updates(
          update_chain: tuple[Callable, ...]
):
     def chained_updates(variable_values: TensorDict, state: GradSolverState):
          for update_fn in update_chain:
               variable_values, state = update_fn(variable_values, state)
          return variable_values, replace(state, iter_=state.iter_ + 1)
     
     return chained_updates

##### Sapphire builder #####
def _sapphire_step_builder(config: SapphireConfig, op_split: SapphireSplit):
     n = op_split.f.dataloader.dataset.num_samples
     conv_factor = n // config.grad_batch_size
     
     if config.base_method == "sgd":
          gradient_fn = core.SGDOracle.build_gradient_fn(op_split)
     elif config.base_method == "svrg":
          update_threshold = n // config.grad_batch_size
          gradient_fn = core.SVRGOracle.build_gradient_fn(op_split, update_threshold=update_threshold)
     elif config.base_method == "saga":
          gradient_fn = core.SAGAOracle.build_gradient_fn(
               op_split, n=n, has_intercept=op_split.f.fit_intercept)
     else:
          raise ValueError(f"Unsupported base_method: {config.base_method}")
     
     device = op_split.f.variable_values.to_flat_tensor().device
     conv_factor = n // config.grad_batch_size
     check_termination_freq = config.check_termination_freq * conv_factor
     precond_update_freq = config.precond_update_freq * conv_factor
          
     loader_fn = _op_split_partial(core.load_batch, op_split)
     prox_fn = _op_split_partial(core.prox_update, op_split)
     termination_fn = _op_split_partial(core.grad_mapping_norm, op_split)
     update_precond_fn = _build_preconditioner_update(
     config.precond_config, op_split, precond_update_freq, device
     )    
     config_cond = isinstance(config.precond_config, IdentityConfig) or \
          (isinstance(config.precond_config, NystromConfig) and op_split.r == [])
          
     if config_cond:
          return  _sapphire_pipeline_chain(
                    loader_fn,
                    gradient_fn,
                    core.precond_grad,
                    prox_fn,
                    update_precond_fn,
                    termination_fn,
                    check_termination_freq=check_termination_freq           
               )
     else:
          prox_P_fn = partial(
                    core.prox_update_P, 
                    op_split=op_split, 
                    subproblem_iters=config.subproblem_iters 
          )
     return _sapphire_pipeline_chain(
                    loader_fn,
                    gradient_fn,
                    None,
                    prox_P_fn,
                    update_precond_fn,
                    termination_fn,
                    check_termination_freq=check_termination_freq
          )


def _sapphire_pipeline_chain(
          loader_fn: Callable,
          gradient_fn: Callable,
          transform_fn: Callable | None,
          update_fn: Callable,
          update_precond_fn: Callable,
          termination_fn: Callable,
          check_termination_freq: int = 100

):
     """Pipeline: gradient → transform → update, with optional preconditioner updates"""

     if transform_fn is None:
          transform_fn = core.identity_transform

     def step(variable_values: TensorDict, state: SapphireState):   
    
        # Get batch
        batch = loader_fn()

        # Update preconditioner if needed
        state = update_precond_fn(batch, variable_values, state)
        
        # Compute gradient
        grads, state = gradient_fn(variable_values, state, batch)
        
        # Transform gradient (e.g., preconditioning)
        updates  = transform_fn(grads, state)
        
        # Apply update
        variable_values, state = update_fn(variable_values, updates, state)

        if state.iter_ % check_termination_freq == 0:
            variable_values, state = termination_fn(
                 variable_values, state               
            )
        
        return variable_values, replace(state, iter_=state.iter_ + 1)

     return step


def _build_preconditioner_update(
    precond_config: PreconditionerConfig,
    op_split: SapphireSplit,
    precond_update_freq: int,
    device: torch.device
) -> tuple[Callable, Callable]:
    """
    Build preconditioner update function.
    
    Returns:
        (update_fn, apply_fn) where:
            - update_fn(iter, beta_value) -> None: updates preconditioner if needed
    """
    if isinstance(precond_config, IdentityConfig):
        # No-op functions for identity preconditioner
        def update_fn(
                  batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                  var_values: TensorDict, 
                  state: GradSolverState        
        ) -> GradSolverState:
            return state
        
        return update_fn
    
    elif isinstance(precond_config, NystromConfig):
        # Create the Nystrom preconditioner
        P = Nystrom(precond_config)
        
        def update_fn(
                  batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                  beta_value: TensorDict, 
                  state: GradSolverState
        )-> GradSolverState:
            if state.iter_ % precond_update_freq == 0:
                X_batch, y_batch, _ = batch
                Hop = op_split.get_subsamp_hessian_linop(
                    beta_value, X_batch, y_batch, device
                )
                P._update(Hop, X_batch.dtype)
                return replace(state, P=P)
            else:
                return state
        
        return update_fn
    
    else:
        raise ValueError(f"Unsupported preconditioner config: {type(precond_config)}")

##### Helpers #####
def _op_split_partial(fn: Callable, op_split: ProxGradSplit | SapphireSplit):
     "Binds op_split fo fn."
     return partial(fn, op_split=op_split)