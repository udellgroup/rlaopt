"""Implementation of the L1-norm atom."""

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


class L1Norm(Atom):
    """L1-norm atom."""

    def __init__(self, x: Variable, scaling: float | torch.Tensor = 1.0):
        """Initializes the L1-norm atom with optional scaling.

        Args:
            x: Variable to apply the L1-norm to.
            scaling: Scaling factor for the L1-norm (default: 1.0).

        Raises:
        TypeError: If x is not a Variable.
        """
        super().__init__(
            exprs={"x": x}, buffers={"scaling": scaling}, variable_names=["x"]
        )

    def is_smooth(self) -> bool:
        """Returns False because L1-norm is not smooth."""
        return False

    def forward(self) -> torch.Tensor:
        """Evaluates the scaled L1-norm."""
        value = self.get_input("x").forward()
        return self.get_buffer("scaling") * torch.sum(torch.abs(value))

    def is_proxable(self) -> bool:
        """Returns True because L1-norm is proxable."""
        return True

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

    def is_subsamplable(self) -> bool:
        """Returns False because L1-norm is not subsamplable."""
        return False

    def subsample(self, indices) -> Self:
        """Raises NotImplementedError because L1-norm cannot be subsampled."""
        raise NotImplementedError("L1-norm cannot be subsampled.")

    def _scale(self, scaling: float) -> Self:
        """Scale the L1-norm atom."""
        new_scaling = self.get_buffer("scaling") * scaling
        return L1Norm(self.get_input("x"), scaling=new_scaling)
