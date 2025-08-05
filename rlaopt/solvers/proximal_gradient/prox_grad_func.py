from math import sqrt

import torch
from typing import Callable, NamedTuple, Optional, Tuple

from ..configs import ProxGradConfig
from ..utils import split_objective
from ...expression.expression import Expression, AddExpression
from ...params.params import Params

class ProxGradState(NamedTuple):
    eta: torch.Tensor
    params_prev: Optional[Params] = None
    err: torch.Tensor = torch.inf
    iter_: int = 0
   
def proximal_gradient(
        obj: Expression | AddExpression,
        config: ProxGradConfig,
        init_params: Optional[Params] = None,

)-> Tuple[Params, torch.Tensor]:

     # Unpack config
     eta, max_iters, tol, use_acceleration, use_linesearch = config.eta, config.max_iters,\
     config.tol, config.use_acceleration, config.use_linesearch
     
     # If no stepsize, set inital stepsize:
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
     else:
        params = obj.params
     state = _init_state(params, eta, use_acceleration)

     # Get error tolerance
     epsilon = tol * sqrt(params.dim)

    # Solver loop
     while state.err > epsilon and state.iter_ <= max_iters:
        params, state = step(params, state)
     return params, state.err

def _init_state(
            params: Params, 
            eta: torch.Tensor,
            use_acceleration: bool 
    )->ProxGradState:
        if use_acceleration:
          return ProxGradState(params_prev=params, eta=eta)
        else:
          return ProxGradState(eta=eta)

def _build_step(
          obj: Expression | AddExpression,  
          use_acceleration: bool, 
          use_linesearch: bool
)->Callable[[Params, ProxGradState, Expression, Callable], Tuple[Params, ProxGradState]]:
     # Get the smooth part of the objective and the prox operator
     f, prox = split_objective(obj)
     
     # f in functional form
     def f_func(params: Params):
          return f.evaluate(params)
     
     # Setup gradient function
     def grad_f(params: Params):
          grads = torch.func.grad(f_func)(params)
          return grads
     
     # Setup function computing stopping criteria
     def err_fn(params: Params, state: ProxGradState)-> torch.Tensor:
          grads = grad_f(params)
          delta_params = params - prox(params - state.eta * grads, state.eta)
          err = delta_params.norm() / state.eta
          return err
     
     if use_acceleration and use_linesearch:
            def step(params: Params, state: ProxGradState)->Tuple[Params, ProxGradState]:
                 params_prev = params
                 params, state = _accel_prox_grad_ls_step(params, state, f_func, grad_f, prox)
                 return params, state._replace(iter_=state.iter_+1, params_prev=params_prev)    
     elif use_acceleration:
             def step(params:Params, state:ProxGradState)->Tuple[Params, ProxGradState]:
                 params_prev = params
                 params = _accel_prox_grad_step(params, state, grad_f, prox)
                 err = err_fn(params, state)
                 return params, state._replace(iter_=state.iter_+1, err=err, params_prev=params_prev)
     elif use_linesearch:
             def step(params:Params, state:ProxGradState)-> Tuple[Params, ProxGradState]:
               params, state = _linesearch(params, state, f_func, grad_f, prox)
               return params, state._replace(iter_=state.iter_+1)  
     else:
            def step(params: Params, state: ProxGradState)->Tuple[Params, ProxGradState]:
                 params = _prox_grad_step(params, state, grad_f, prox)
                 err = err_fn(params, state)
                 return  params, state._replace(iter_=state.iter_+1, err=err)
     return step

def _prox_grad_step(
            params: Params,
            state: ProxGradState,
            grad_f: Callable[[Params], Params],
            prox: Callable[[Params, float], Params]
    ):  
        grads = grad_f(params)
        params = _prox_update(params, grads, state, prox)
        return params

def _accel_prox_grad_ls_step(params: Params,
                             state: ProxGradState,
                             f: Callable[[Params], torch.Tensor],
                             grad_f: Callable[[Params], Params],
                             prox: Callable[[Params, float], Params]):
     
     y = _accel_step(params, state)
     params, state = _linesearch(y, state, f, grad_f, prox)
     return params, state
    
def _accel_prox_grad_step(
            params: Params,
            state: ProxGradState,
            grad_f: Callable[[Params], Params],
            prox: Callable[[torch.Tensor, float], torch.Tensor]         
    ):  
        y = _accel_step(params, state)
        params = _prox_grad_step(y, state, grad_f, prox)
        return params

def _accel_step(params: Params, state: ProxGradState):
     return params + state.iter_ / (state.iter_ + 3) * (params - state.params_prev)

def _linesearch(
          params: Params, 
          state: ProxGradState,
          f: Callable[[Params], torch.Tensor],
          grad_f: Callable[[Params], Params],
          prox: Callable[[Params, float], Params]
     )->Tuple[Params, ProxGradState]:

     beta = 0.5
     f0 = f(params)
     grads = grad_f(params)
     cond = False

     def linesearch_step(params: Params, state: ProxGradState):
         z = _prox_update(params, grads, state, prox)
         d = z - params
         u = f0 + grads.dot(d) + 1 / (2 * state.eta) * (d.norm()) ** 2
         if f(z) <= u:
              err_new = d.norm() / state.eta
              return True, z, state._replace(err=err_new)
         else:
              eta_new = beta * state.eta
              return False, params, state._replace(eta=eta_new)

     while not cond:
          cond, params, state = linesearch_step(params, state)
     
     return params, state

def _prox_update(
          params: Params, 
          grads: Params, 
          state: ProxGradState, 
          prox: Callable[[Params, float], Params])-> Params:
     return prox(params - state.eta * grads, state.eta)