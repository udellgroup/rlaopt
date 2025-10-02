import torch

from rlaopt.atoms.box import Polyhedra
from rlaopt.expression.expression import Variable

class Halfspace(Polyhedra):
    def __init__(self, x: Variable, c : torch.Tensor , u: torch.Tensor):
        super().__init__(x, A=None, b=None, C=c, u=u)
    
    def is_proxable(self):
        return True
    
    def prox(self, location: torch.Tensor, prox_scaling: float):
        c_norm = torch.linalg.norm(self.c, 2)
        r = torch.dot(self.c, location) - self.u
        zero = torch.tensor(0.0, device=r.device)
        return location - torch.maximum(r, zero) * location / c_norm ** 2