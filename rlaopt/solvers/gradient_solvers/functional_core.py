from abc import ABC, abstractmethod
from dataclasses import replace
from functools import partial
from typing import Any, Callable

import torch

from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.prox_grad_split import ProxGradSplit
from rlaopt.splitting.sapphire_split import SapphireSplit
from .optim_states import (GradSolverState, ProxGradState, SapphireState, SGDState, SAGAState, SVRGState)

##### Gradient Oracles #####
def full_gradient(
            var_vals: TensorDict, 
            op_split: ProxGradSplit
)->TensorDict:
      return op_split.grad_f(var_vals)

class StochasticGradientOracle(ABC):
     
     @classmethod
     def build_gradient_fn(
          cls, op_split: SapphireSplit, **kwargs: Any
     )->Callable[[TensorDict, SapphireState, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 tuple[TensorDict, SapphireState]]:
          return partial(cls.gradient, op_split=op_split, **kwargs)
     
     @staticmethod
     @abstractmethod
     def gradient(
          var_vals: TensorDict,
          state: SapphireState,
          batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
          op_split: SapphireSplit,
          **kwargs: Any
        )->TensorDict:
          pass

class SGDOracle(StochasticGradientOracle):
     
     @staticmethod
     def gradient(
          var_vals: TensorDict, 
          state: SGDState, 
          batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
          op_split: SapphireSplit
    ):
          X_batch, y_batch, _ = batch
          return op_split.batch_loss_grad(var_vals, X_batch, y_batch), state

class SVRGOracle(StochasticGradientOracle):
     
     @staticmethod
     def gradient(
          var_vals: TensorDict, 
          state: SVRGState, 
          batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
          op_split: SapphireSplit, 
          update_threshold: int
    ):
          X_batch, y_batch, _ = batch
          # Update snapshot if needed
          if state.iter_ % update_threshold == 0:
                grad_snapshot = full_gradient(var_vals, op_split)
                state = replace(state, snapshot=var_vals, snapshot_grad=grad_snapshot)
        
          # Compute SVRG gradient
          X_batch, y_batch, _ = load_batch(op_split)
          v = (op_split.batch_loss_grad(var_vals, X_batch, y_batch) - 
                op_split.batch_loss_grad(state.snapshot, X_batch, y_batch) + 
                state.snapshot_grad
            )
          return v, state

class SAGAOracle(StochasticGradientOracle):
     
     @staticmethod
     def gradient(
          var_vals: TensorDict, 
          state: SAGAState, 
          batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
          op_split: SapphireSplit, 
          n: int, 
          has_intercept: bool
    ):
        #  Extract data tensors and batch indices
        X_batch, y_batch, batch_indices = batch
        batch_size = X_batch.shape[0]
  
        # Get table weights for current batch
        tbl_weights = state.table[batch_indices]
        
        # Compute prediction on batch
        y_pre = X_batch @ var_vals['beta']
        if has_intercept:
            y_pre = y_pre + var_vals['intercept']
        
        # Compute new table weights
        op_split.f._loss_fn.reduction = "sum"
        new_weights = torch.func.grad(op_split.f._loss_fn)(y_pre, y_batch)       
        op_split.f._loss_fn.reduction = "mean"
        
        # Get aux tensor for beta coefficients
        aux_beta = X_batch.T @ (new_weights - tbl_weights)
    
        # Get aux tensor for intercept (if present)
        if has_intercept:
            aux_intercept = (new_weights - tbl_weights).sum()
            aux = TensorDict({'beta': aux_beta, 'intercept': aux_intercept})
        else:
            aux = TensorDict({"beta": aux_beta})
       
        # Get new gradient
        v = state.grad_avg + 1 / batch_size * aux

        # Update the table and average
        grad_avg_new = state.grad_avg + 1 / n * aux

        table_new = state.table.clone()
        table_new[batch_indices] = new_weights    
    
        return v, replace(state, table=table_new, grad_avg=grad_avg_new)

##### Transform functions #####
def identity_transform(grads: TensorDict, state: SapphireState):
                  return grads

def precond_grad(
            grads: TensorDict,
            state: SapphireState
)-> TensorDict:
      return grads.from_flat_tensor(state.P.inv @ grads.to_flat_tensor())

##### Update functions #####
def nest_accel_update(
          var_vals: TensorDict, 
          state: ProxGradState
) -> tuple[TensorDict, ProxGradState]:
     """Compute the accelerated (momentum) iterate."""
     var_vals_prev = var_vals.clone()
     momentum_scale = state.iter_ / (state.iter_ + 3)
     return var_vals + momentum_scale * (var_vals - state.variable_values_prev), replace(state, variable_values_prev=var_vals_prev)


def prox_update(
          var_vals: TensorDict,
          updates: TensorDict, 
          state: ProxGradState, 
          op_split: ProxGradSplit
):
       return op_split.prox(var_vals - state.eta * updates, state.eta), state


def prox_update_P(
          var_vals: TensorDict,
          grads: TensorDict,
          state: SapphireState,
          op_split: SapphireSplit,
          subproblem_iters: int
):
     alpha = 1 / state.P.norm
     t_old = 1.0
     z0 = var_vals.clone()
     y = z0
     z_old = z0
     
     for _ in range(subproblem_iters):
        Pd = y.from_flat_tensor(state. P @ (y - var_vals).to_flat_tensor())   
        y_int = y - alpha * (state.eta * grads + Pd)

        z0 = op_split.prox(y_int, state.eta * alpha)
        t = (1 + (1 + 4 * t_old**2) ** (0.5)) / 2
        y = z0 + (t_old - 1) / t * (z0 - z_old)

        t_old = t
        z_old = z0

     return y, state
     

def load_batch(op_split: SapphireSplit):
      return op_split.f.dataloader.get_batch()


##### Step functions #####
def prox_gd_step(
          var_vals: TensorDict,
          state: ProxGradState,
          op_split):
     grads = full_gradient(var_vals, op_split)
     return prox_update(var_vals, grads, state, op_split)  

def linesearch(
    var_vals: TensorDict,
    state: ProxGradState,
    op_split: ProxGradSplit
) -> tuple[TensorDict, ProxGradState]:
        """Perform backtracking line search to find appropriate step size."""
        beta = 0.5
        f0 = op_split.func_f(var_vals)
        grads = full_gradient(var_vals, op_split)
        cond = False

        def linesearch_step(var_vals: TensorDict, state: ProxGradState):
            z, _ = prox_update(var_vals, grads, state, op_split)
            d = z - var_vals
            u = f0 + grads.flat_dot(d) + 1 / (2 * state.eta) * (d.flat_norm() ** 2)

            if  op_split.func_f(z) <= u:
                err_new = d.flat_norm() / state.eta
                return True, z, replace(state, err=err_new)
            else:
                eta_new = beta * state.eta
                return False, var_vals, replace(state, eta=eta_new)

        while not cond:
            cond, var_vals, state = linesearch_step(var_vals, state)

        return var_vals, state 


#### Error metrics ####
def grad_mapping_norm(
          var_vals: TensorDict, state: GradSolverState, op_split: ProxGradSplit | SapphireSplit
)->float:
     var_vals_new, _ = prox_gd_step(var_vals, state, op_split)
     G = (var_vals - var_vals_new) / state.eta
     return var_vals, replace(state, err=G.flat_norm().item())