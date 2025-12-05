"""Implementation of the L2-norm atom."""

import torch

from rlaopt.atoms.nonsmooth_regularizer import NonsmoothRegularizer
from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict


class L2Norm(NonsmoothRegularizer):
    """L2-norm regularization atom.
    
    Computes the scaled L2-norm: scaling * ||x||₂ = scaling * sqrt(Σᵢ xᵢ²)

    Args:
        x: Expression to apply the L2-norm to.
        scaling: Scaling factor for the L2-norm (default: 1.0).
        
    Examples:
        >>> x = Variable((100,), name='weights')
        >>> l2 = L2Norm(x, scaling=0.01)
        >>> penalty = l2.forward()
    """
    
    def __init__(self, x: Expression, scaling: float | torch.Tensor = 1.0):
        """Initializes the L2-norm atom with optional scaling.

        Args:
            x: Expression to apply the L2-norm to.
            scaling: Scaling factor for the L2-norm (default: 1.0).
        """
        super().__init__(x, scaling_params={"scaling": scaling})

    def forward(self) -> torch.Tensor:
        """Evaluates the scaled L2-norm."""
        value = self.get_input("x").forward()
        return self.get_buffer("scaling") * torch.sqrt(torch.sum(value**2))
    
    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Compute the proximal operator of the scaled L2-norm.

        For f(x) = scaling * ||x||_2 and prox parameter tau = prox_scaling, the
        proximal operator is:

            prox_{τ f}(v) = (1 - scaling * tau / ||v||_2)_+ * v

        where (·)_+ = max(·, 0). If ||v||_2 <= scaling * tau, then prox(v) = 0.
        """
        scaling = self.get_buffer("scaling")
        lam = scaling * prox_scaling 

        def prox_l2(x: torch.Tensor) -> torch.Tensor:
            norm = torch.linalg.norm(x, p=2)
            if norm == 0:
                return torch.zeros_like(x)
            factor = torch.clamp(1.0 - lam / norm, min=0.0)
            return factor * x

        return relevant_variable_values.apply(prox_l2)