import torch

from rlaopt.atoms.polyhedra import Polyhedra
from rlaopt.expression.expression import Variable

class Box(Polyhedra):
    def __init__(self, x: Variable, l: torch.Tensor=None, u: torch.Tensor=None):
        super().__init__(x, A=None, b=None, C=None, l=l, u=u)
    
    def is_proxable(self):
        return True
    
    def prox(self, location: torch.Tensor, prox_scaling: float)->torch.Tensor:
        return torch.clamp(location, self.l, self.u)
