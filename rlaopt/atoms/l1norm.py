"""Implementation of the L1-norm atom."""

import cvxpy as cp
import torch

from rlaopt.atoms.atom import Atom
from rlaopt.variable import Variable


class L1Norm(Atom):
    """L1-norm atom with scaling: scaling * ||x||_1"""

    def __init__(self, x: Variable, scaling: float | torch.Tensor = 1.0):
        """Initializes the L1-norm atom with optional scaling.

        Args:
            x: Variable to apply the L1-norm to.
               Must be an instance of rlaopt.variable.Variable.
            scaling: Scaling factor for the L1-norm (default: 1.0).
                     Can be a float or a torch.Tensor.
        """
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        # Register the variable as a parameter
        self.register_parameter("x", x)

        # Register the scaling factor
        if isinstance(scaling, torch.Tensor):
            self.register_buffer("scaling", scaling)
        else:
            self.register_buffer("scaling", torch.tensor(float(scaling)))

    def forward(self, location=None) -> torch.Tensor:
        """Evaluation of the scaled L1-norm: scaling * ||x||_1

        Args:
            location: Optional tensor at which to evaluate the L1-norm.
                     If provided, uses this location instead of the stored variable.

        Returns:
            Scaled sum of absolute values (L1-norm) of the input
        """
        if location is not None:
            return self.scaling * torch.sum(torch.abs(location))
        return self.scaling * torch.sum(torch.abs(self.x))

    def is_smooth(self) -> bool:
        """Returns False because L1-norm is not smooth."""
        return False

    def is_proxable(self) -> bool:
        """Returns True because L1-norm is proxable."""
        return True

    def prox(self, location, prox_scaling) -> torch.Tensor:
        """Proximal operator for the scaled L1-norm.

        For the function f(x) = scaling * ||x||_1, the proximal operator is:
        prox_f(v) = sign(v) * max(|v| - scaling * prox_scaling, 0)

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Additional scaling factor for the proximal operator

        Returns:
            Result of soft-thresholding operation
        """
        threshold = self.scaling * prox_scaling
        return torch.sign(location) * torch.clamp(
            torch.abs(location) - threshold, min=0.0
        )

    def is_subsamplable(self) -> bool:
        """Returns False because L1-norm is not subsamplable."""
        return False

    def subsample(self, indices) -> "L1Norm":
        """Raises NotImplementedError because L1-norm cannot be subsampled."""
        raise NotImplementedError("L1-norm cannot be subsampled.")

    def to_cvxpy(self, expr=None) -> cp.Expression:
        """Converts the scaled L1-norm to a CVXPY expression.

        Args:
            expr: Optional CVXPY expression to use.
                 If provided, uses this expression instead of converting
                 the stored variable.

        Returns:
            CVXPY expression for the scaled L1-norm
        """
        scaling_value = float(self.scaling.item())

        if expr is not None:
            return scaling_value * cp.norm(expr, 1)

        # Convert the stored variable to a CVXPY expression
        return scaling_value * cp.norm(self.x.to_cvxpy(), 1)
