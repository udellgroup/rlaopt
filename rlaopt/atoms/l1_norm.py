"""Implementation of the L1-norm atom."""

import torch

from rlaopt.atoms.nonsmooth_regularizer import NonsmoothRegularizer
from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict


class L1Norm(NonsmoothRegularizer):
    """L1-norm regularization atom.

    Computes the scaled L1-norm: scaling * ||x||₁ = scaling * Σᵢ |xᵢ|

    Args:
        x: Expression to apply the L1-norm to.
        scaling: Scaling factor for the L1-norm (default: 1.0).

    Examples:
        >>> x = Variable((100,), name='weights')
        >>> l1 = L1Norm(x, scaling=0.01)
        >>> penalty = l1.forward()
    """

    def __init__(self, x: Expression, scaling: float | torch.Tensor = 1.0):
        """Initializes the L1-norm atom with optional scaling.

        Args:
            x: Expression to apply the L1-norm to.
            scaling: Scaling factor for the L1-norm (default: 1.0).
        """
        super().__init__(x, scaling_params={"scaling": scaling})

    def forward(self) -> torch.Tensor:
        """Evaluates the scaled L1-norm."""
        value = self.get_input("x").forward()
        return self.get_buffer("scaling") * torch.sum(torch.abs(value))

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Compute the proximal operator of the scaled L1-norm.

        For the function f(x) = scaling * ||x||_1, the proximal operator is:
        prox_f(v) = sign(v) * max(|v| - scaling * prox_scaling, 0)
        """
        scaling = self.get_buffer("scaling")
        threshold = scaling * prox_scaling

        def soft_threshold(x: torch.Tensor) -> torch.Tensor:
            return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - threshold)

        return relevant_variable_values.apply(soft_threshold)
