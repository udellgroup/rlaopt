"""Implementation of the L1-norm atom."""

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Variable


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

    def prox(self, location, prox_scaling) -> torch.Tensor:
        """Proximal operator for the scaled L1-norm.

        For the function f(x) = scaling * ||x||_1, the proximal operator is:
        prox_f(v) = sign(v) * max(|v| - scaling * prox_scaling, 0)

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Scaling factor for the proximal operator

        Returns:
            Result of soft-thresholding operation
        """
        threshold = self.get_buffer("scaling") * prox_scaling
        return torch.nn.functional.relu(
            location - threshold
        ) - torch.nn.functional.relu(-location - threshold)

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
