import cvxpy as cp
import torch

from .atom import Atom
from ..expression.variable import Variable

class AffineAtom(Atom):
    def __init__(self, x: Variable, A: torch.Tensor, b: torch.Tensor):
        super().__init__()

        if not isinstance(x, (Variable)):
            raise TypeError(f"Expected Variable, got {type(x)}")
        
        self.register_parameter("x", x.value)
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def is_smooth(self):
        return True
    
    def evaluate_at(self, **variable_locations):
        return self.A @ self.x + self.b
    
    def is_proxable(self):
        return False
    
    def prox(self, location, prox_scaling):
        return super().prox(location, prox_scaling)
    
    def is_subsamplable(self):
        return True
    
    def subsample(self, indices) -> "AffineAtom":
        return AffineAtom(self.x, self.A[indices], self.b[indices])
    
    def to_cvxpy(self):
        return super().to_cvxpy()