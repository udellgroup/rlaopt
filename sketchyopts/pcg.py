import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax.lax import while_loop
from sketchyopts.linop import ExtendedLinearOperator
from sketchyopts import sketches
from sketchyopts.lra import rand_nystrom_approx
from typing import Any, Callable, NamedTuple, Tuple, Union

class PCG_State(NamedTuple):
    params: jnp.ndarray
    r: jnp.ndarray
    p: jnp.ndarray
    z: jnp.ndarray
    r_dot_z: jnp.ndarray
    error: float = jnp.inf
    iter_count: int = 0

def batch_pcg(init_params: jnp.ndarray, 
              A: Union[jnp.ndarray, ExtendedLinearOperator], 
              B: jnp.ndarray, 
              precond_params: dict, 
              damp: float,
              is_array: bool,
              is_square: bool, 
              max_iter: int,
              tol: float,
              *args: Any
              ):
     _pcg = lambda p, b: pcg(p, A, b, precond_params, damp, is_array, is_square, max_iter, tol, *args)
     return vmap(_pcg)(init_params, B)

def pcg(init_params: jnp.ndarray, 
        A: Union[jnp.ndarray, ExtendedLinearOperator], 
        b: jnp.ndarray, 
        precond_params: dict, 
        damp: float,
        is_array: bool,
        is_square: bool, 
        max_iter: int, 
        tol: float,
        *args: Any
        ):
    Hop, P, P_params = _init_pcg_precond(A, precond_params, damp, is_array, is_square, *args)
    if is_square:
        if is_array:
            state = _pcg(init_params, Hop, b, P, tol, max_iter, P_params, A)  
        else:
            state = _pcg(init_params, Hop, b, P, tol, max_iter, P_params, *args)
    else:
        AT_b = A.T@b
        if is_array:
           state = _pcg(init_params, Hop, AT_b, P, tol, max_iter, P_params, A)  
        else:
         state = _pcg(init_params, Hop, AT_b, P, tol, max_iter, P_params, *args)
    
    return state 

def _init_pcg_precond(A: Union[jnp.ndarray, ExtendedLinearOperator], 
                      precond_params: dict, 
                      damp: float,
                      is_array: bool,
                      is_square: bool,
                      *args: Any
                     )->Tuple[Callable, Callable, Tuple[jnp.ndarray, jnp.ndarray]]:
    
    ###### Setup callable for _pcg function ###### 
    p = A.shape[1]
    if is_square:
        if is_array:
            def Hop(v: jnp.ndarray, A:jnp.ndarray)->jnp.ndarray: # type: ignore
                return A@v+damp*v
        else:
            def Hop(v: jnp.ndarray, *args: Any)->jnp.ndarray:
                return A.matvec(v,*args)+damp*v # type: ignore

        if precond_params['type'] == 's&p':
          raise ValueError("s&p preconditioner is not supported for square linear systems. Please use nystrom instead.") 
    else:
        if is_array:
            def Hop(v: jnp.ndarray, A:jnp.ndarray)->jnp.ndarray: # type: ignore
                return A.T@(A@v)+damp*v
        else:
            def Hop(v: jnp.ndarray, *args: Any)->jnp.ndarray: 
                return A.rmatvec(A.matvec(v,*args),*args)+damp*v # type: ignore

    ###### Setup preconditioner callable for _pcg function ######  
    if precond_params['type'] == 'nystrom':
         U, S = rand_nystrom_approx(A, precond_params['sketch_size'], precond_params['key'], precond_params['sketch_type'], is_array, *args)
         def _apply_precond(v: jnp.ndarray, precond: Tuple[jnp.ndarray, jnp.ndarray])->jnp.ndarray: # type: ignore
             UTv = precond[0].T@v
             return precond[0]@(jnp.divide(UTv, precond[1]+damp))+1/damp*(v-precond[0]@(UTv))
         return Hop, _apply_precond, (U,S)
    else:
        n,p = A.shape
        if precond_params['sketch_type'] == 'gauss':
            S, _ = sketches.gauss_sketch_mat(precond_params['key'], (n,p), precond_params['sketch_size'], 'left')
        elif precond_params['sketch_type'] == 'ortho':
            S, _ = sketches.ortho_sketch_mat(precond_params['key'], (n,p), precond_params['sketch_size'], 'left') 
        else:
            S, _ = sketches.sjlt_sketch_mat(precond_params['key'], (n,p), precond_params['sketch_size'], 'left') 
        if is_array:
            SA = S@A
        else:
            SA = (A.rmatmul(S.T, *args)).T # type: ignore

        if precond_params['sketch_size']<=p:
                L = jsp.linalg.cholesky(SA@SA.T+damp*jnp.eye(precond_params['sketch_size']), lower=True)
                def _apply_precond(v: jnp.ndarray, precond_params: Tuple[jnp.ndarray, jnp.ndarray])->jnp.ndarray:
                    temp = precond_params[1]@v
                    temp = jsp.linalg.solve_triangular(precond_params[0], temp, trans=0, lower=True)
                    temp = jsp.linalg.solve_triangular(precond_params[0], temp, trans=1, lower=True)
                    Pv = 1/damp*(v-precond_params[1].T@temp)
                    return Pv
                return Hop, _apply_precond, (L, SA)
        else:
            L = jsp.linalg.cholesky(SA.T@SA+damp*jnp.eye(p), lower=True)
            def _apply_precond(v: jnp.ndarray, precond_params: Tuple[jnp.ndarray, jnp.ndarray])->jnp.ndarray:
                temp = jsp.linalg.solve_triangular(precond_params[0], v, trans=0, lower=True)
                Pv = jsp.linalg.solve_triangular(precond_params[0], temp, trans=1, lower=True)
                return Pv
        return Hop, _apply_precond, (L, SA)


def _pcg(init_params: jnp.ndarray, 
         Hop: Callable[[jnp.ndarray, Tuple[Any]], jnp.ndarray], 
         b: jnp.ndarray, 
         P: Callable[[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray], 
         tol: float, 
         max_iter:int,
         P_params: Tuple[jnp.ndarray, jnp.ndarray], 
         *args: Any)->PCG_State:
    
    def _init_pcg():
        r = b-Hop(init_params, *args)
        z = P(r, P_params)
        p = jnp.copy(z)
        r_dot_z = jnp.dot(z,r)
        return PCG_State(params=init_params, r=r, p=p, z=z, r_dot_z=r_dot_z)
    
    def _body_fun(state: PCG_State):
       Hp = Hop(state.p, *args)
      
       # Update solution and residual
       alpha = state.r_dot_z / jnp.dot(Hp, state.p)
       new_params = state.params+alpha * state.p
       r = state.r-alpha * Hp
       new_error = jnp.linalg.norm(r)/jnp.linalg.norm(b)
       # Apply preconditioner
       z = P(r, P_params)

      # Update search direction
       rnp1_dot_znp1 = jnp.dot(r, z)
       p = z + (rnp1_dot_znp1 / state.r_dot_z) * state.p
       return state._replace(params=new_params, 
                             r=r, 
                             p=p, 
                             z=z, 
                             r_dot_z=rnp1_dot_znp1, 
                             error=new_error, 
                             iter_count=state.iter_count+1)
    
    def _cond_fun(state: PCG_State)->bool:
       return (state.error > tol) & (state.iter_count < max_iter)

    state = _init_pcg()

    # Optimization loop  
    state = while_loop(_cond_fun, _body_fun, state)
    
    return state