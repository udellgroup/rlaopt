import torch

from rlaopt.atoms.polyhedra import Polyhedra
from rlaopt.expression.expression import Variable

class AffineConstraint(Polyhedra):
    def __init__(self, x: Variable, A: torch.Tensor, b: torch.Tensor):
        super().__init__(x, A=A, b=b, C=None, l=None, u=None)
        self._prox = _build_prox(A, b, prox_mode="exact")

    def is_proxable(self):
        return True
    
    def prox(self, location: torch.Tensor, prox_scaling: float):
        return self._prox(location, prox_scaling)

def _build_prox(A: torch.Tensor, b:torch.Tensor, prox_mode: str):
    if prox_mode == "exact":

       def prox(location, prox_scaling: float):
           r = A @ location - b
           G = A @ A.T 
           L = torch.linalg.cholesky(G)

           # G^(-1)r
           temp = torch.linalg.solve_triangular(L.T, 
                                                torch.linalg.solve_triangular(L, r.reshape(r.shape[0], 1), upper=False),
                                                upper=True
           )
           return location - A.T @ temp.reshape(temp.shape[0],)
    
    return prox   