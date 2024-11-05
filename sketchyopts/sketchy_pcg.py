import jax.numpy as jnp
import jax.random as jrn
from dataclasses import dataclass
from sketchyopts.linop import ExtendedLinearOperator
from typing import Any, Union
from sketchyopts.pcg import pcg, batch_pcg

@dataclass(eq=False)
class SketchyPCG:
    precond_type: str = 'nystrom'
    sketch_type: str = 'sjlt'
    max_iter: int = 200
    tol: float = 10**-6
    seed: int = 0

    def __post_init__(self):
        if self.precond_type not in ['nystrom', 's&p']:
           raise ValueError(f"We do not support the preconditioner: {self.precond_type}")
        if self.sketch_type not in ['gauss', 'ortho', 'sjlt']:
           raise ValueError(f"We do not support the sketch: {self.sketch_type}") 

    def run(self,
        init_params: jnp.ndarray, 
        A: Union[jnp.ndarray, ExtendedLinearOperator], 
        B: jnp.ndarray, 
        reg: float, 
        sketch_size: int,
        *args: Any):
    
        n, p = A.shape
        key = jrn.key(self.seed)
        precond_params = dict(type=self.precond_type,sketch_type=self.sketch_type, sketch_size = sketch_size, key=key)
        if isinstance(A, jnp.ndarray):
           is_array = True
        else:
           is_array = False 
        if n==p:
            is_square = True 
        else:
            is_square = False 
        if len(B.shape) == 1:
            state = pcg(init_params, A, B, precond_params, reg, is_array, is_square, self.max_iter, self.tol,*args)
        else:
            state = batch_pcg(init_params, A, B, precond_params, reg, is_array, is_square, self.max_iter, self.tol, *args)
    
        return state