import jax.numpy as jnp
import jax.random as jrn
import jax.scipy as jsp
from jax import grad, Array
from jax.lax import while_loop
from sketchyopts.sketches import gauss_sketch, sjlt_sketch
from typing import Any, Callable, NamedTuple, Tuple
from functools import partial
from dataclasses import dataclass

KeyArray = Array

class NSketch_State(NamedTuple):
    params: jnp.ndarray
    key: Array
    error: float = jnp.inf
    iter_count: int = 0 


@dataclass(eq=False)
class NSketch:
    r""" Newton Sketch solver from Pilanci and Wainwright (2017)

        The Newton Sketch solves problems of the form:

        minimize f(Aw)+reg/2*||w||_2^2,  f is smooth, A in R^(n x p) and reg>=0.

        This method is ideal for the regime when n>>p and p<=3000 (or when nu>0 and the effective dimension of the Hessian is <3000).
        For p outside this range, use a PROMISE solver instead.

        <u> Attributes </u>:

        loss_fun (Callable): a smooth function of the form fun(x, *args)

        hess_sqrt_fun (Callable): a function of the form hess_sqrt_fun(x, *args) that returns a matrix R that is a hessian sqrt for the unregularized portion of the Hessian, i.e. hess(f(Aw)) = R.T@R
        
        sketch_type (str): a string that specifies what type of sketching matrix (gauss, sjlt, srht, row) to use.
        
        tol (float): tolerance for terminating the solver. NSketch will terminate if:

        jnp.linalg.norm(params_new-params_old)<=tol*jnp.linalg.norm(params_old). 

        max_iter: maximum number of NSketch iterations to run 

        seed (int): integer specifying the random seed to use.

    """
    loss_fun: Callable[[jnp.ndarray, tuple[Any, ...]], jnp.ndarray]
    hess_sqrt_fun: Callable[[jnp.ndarray, tuple[Any, ...]], jnp.ndarray]
    sketch_type: str = 'sjlt'
    tol: float = 10**-3
    max_iter: int = 100
    seed: int = 0

    def run(self,
            init_params: jnp.ndarray,
            reg: float, 
            sketch_size: int, 
            *args: Any,
           )->NSketch_State:
        
        r""" Run function for the Newton Sketch.

        init_params (jnp.ndarray): initial value of the optimization variables.

        reg (float): Value of the l2_regularization parameter

        sketch_size (int): specifies the number of rows in the sketching matrix.

        *args (Any): tuple of additional input arguments taken in by self.loss_fun.

        """
    
        p = init_params.shape[0]
        key = jrn.key(self.seed)

        S = _init_sketch(sketch_size, self.sketch_type)
        grad_fun = grad(self.loss_fun)
        
        if reg>0:
           damp = reg
        else:
           damp = jnp.finfo(jnp.float32).eps
        
        if sketch_size>=p:
            _get_search_dir = partial(_inv_appx_hvp, sketch_size=sketch_size, mat_shape='tall')
        else:
             _get_search_dir = partial(_inv_appx_hvp, sketch_size=sketch_size, mat_shape='short')

        
        # Intialize the optimization state
        state = NSketch_State(params=init_params, key=key)

        def _cond_fun(state: NSketch_State)->bool:
            return (state.error > self.tol) & (state.iter_count < self.max_iter)

        def _body_fun(state: NSketch_State)->NSketch_State:
            g = grad_fun(state.params, *args)
        
            R = self.hess_sqrt_fun(state.params, *args)
            SR, key = S(state.key,R) 

            Pg = _get_search_dir(SR, g, damp, p)
        
            alpha = _line_search(self.loss_fun, state.params, g, Pg, *args)
        
            new_params = state.params-alpha*Pg
            new_error = jnp.linalg.norm(new_params-state.params)/jnp.linalg.norm(state.params)
        
            return state._replace(params=new_params, key=key, error=new_error, iter_count=state.iter_count+1)

        # Optimization loop  
        state = while_loop(_cond_fun, _body_fun, state)
     
        return state


def _inv_appx_hvp(SR: jnp.ndarray, 
                  g: jnp.ndarray, 
                  damp: float, 
                  p: int,
                  sketch_size: int, 
                  mat_shape: str)->jnp.ndarray:
     
     if mat_shape == 'tall':
        G = SR.T@SR+damp*jnp.eye(p)
        Pg = jsp.linalg.solve(G, g, assume_a="pos")
     else:
         G = SR@SR.T+damp*jnp.eye(sketch_size)
         temp = SR@g
         temp = jsp.linalg.solve(G,temp,assume_a="pos")
         Pg = 1/damp*(g-SR.T@temp)
     return Pg

def _init_sketch(s: int, sketch_type: 'str')->Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    if sketch_type == 'gauss':
       S = partial(gauss_sketch, s=s, mode='left')    
    elif sketch_type == 'sjlt':
       S = partial(sjlt_sketch, s=s, mode='left')  
    else:
        raise ValueError(f"We do not support the sketch type: {sketch_type}") 
    return S
     

def _line_search(loss_fun: Callable[[jnp.ndarray, Tuple[Any,...]], jnp.ndarray], 
                 params: jnp.ndarray, 
                 grad: jnp.ndarray, 
                 dir: jnp.ndarray,
                 *args: Any, 
                 initial_step_size: float=1.0, 
                 c: float=0.1, 
                 tau: float=0.5, 
                 max_iters: int=10,
                 )->float:
    """
    Perform Armijo line search to find a step size that satisfies the Armijo condition.

    Parameters:
    - loss_fun: Callable, the loss function to minimize.
    - params: jnp.ndarray, current parameters.
    - grad: jnp.ndarray, gradient of the loss at `params`.
    - initial_step_size: float, initial guess for the step size.
    - c: float, Armijo condition constant (0 < c < 1).
    - tau: float, step size reduction factor (0 < tau < 1).
    - max_iters: int, maximum number of iterations.

    Returns:
    - alpha: the step size that satisfies the Armijo condition.
    """
    
    def condition(state)->bool:
        alpha, _, loss_new, iter_count = state
        # Stop if either the Armijo condition is satisfied or max iterations are reached
        return (loss_new > loss_initial - c * alpha * jnp.dot(grad, dir)) & (iter_count < max_iters)
    
    def body(state):
        alpha, params_new, _, iter_count = state
        alpha = tau * alpha  # Reduce step size
        params_new = params - alpha * dir  # Compute new parameters
        loss_new = loss_fun(params_new, *args)  # Compute new loss
        return alpha, params_new, loss_new, iter_count + 1
    
    # Initial setup
    alpha = initial_step_size
    loss_initial = loss_fun(params, *args)
    params_new = params - alpha * dir
    loss_new = loss_fun(params_new, *args)
    iter_count = 0
    
    # Run while loop with the initial state
    alpha, _, _, _ = while_loop(condition, body, (alpha, params_new, loss_new, iter_count))
    return alpha


    
        