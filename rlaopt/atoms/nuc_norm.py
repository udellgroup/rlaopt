from __future__ import annotations

import cvxpy as cp
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable
from rlaopt.atoms._utils import get_variable_value

class NucNorm(AtomExpression):

    def __init__(self, x: Variable, scaling: float | torch.Tensor | torch.nn.Parameter = 1.0):
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")
        
        if x.value.data.dim() != 2:
            raise ValueError(f"Variable value must be 2D Tensor, but got {x.value.data.dim()}D Tensor.")
        
        # Register the variable as a parameter
        self.register_parameter("x", x.value)

        # Register the scaling factor
        if isinstance(scaling, torch.Tensor):
            self.register_buffer("scaling", scaling)
        elif isinstance(scaling, torch.nn.Parameter):
            self.register_buffer("scaling", scaling.data)
        else:
            self.register_buffer("scaling", torch.tensor(float(scaling)))
        
    def is_smooth(self):
            return False
    
    def evaluate_at(self, **variable_locations):
         value = get_variable_value(self.x, **variable_locations) 
         S = torch.linalg.svdvals(value)
         return self.scaling * torch.sum(S)
    
    def is_proxable(self):
         return True
    
    def prox(self, location, prox_scaling):
         U, S, Vt = torch.linalg.svd(location, full_matrices=False)
         S = torch.nn.functional.relu(S - prox_scaling * self.scaling)
         return U @ (S[:, None] * Vt)
    
    def is_subsamplable(self):
         return False
    
    def subsample(self, indices):
         raise NotImplementedError("Nuclear norm cannot be subsampled")
    
    def to_cvxpy(self):
         raise NotImplementedError("NucNorm does not support conversion to cvxpy")