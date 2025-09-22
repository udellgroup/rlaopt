"""Implementation of the L1-norm atom."""

import cvxpy as cp
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable
from rlaopt.atoms._utils import get_variable_value


class L1Norm(AtomExpression):
    """L1-norm atom."""

    def __init__(self, x: Variable, scaling: float = 1.0):
        """Initializes the L1-norm atom with optional scaling.

        Args:
            x: Variable to apply the L1-norm to.
               Must be an instance of Variable.
            scaling: Scaling factor for the L1-norm (default: 1.0).
                     Can be a float or a torch.Tensor.
        """
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        # Register the variable as a parameter
        self.register_parameter("x", x.value)

        # Register the scaling factor
        if isinstance(scaling, torch.Tensor):
            self.register_buffer("scaling", scaling)
        elif isinstance(scaling, torch.nn.Parameter):
            self.register_buffer("scaling", scaling.data)
        else:
            self.register_buffer("scaling", torch.tensor(float(scaling)))

    def is_smooth(self) -> bool:
        """Returns False because L1-norm is not smooth."""
        return False

    def evaluate_at(self, **variable_locations):
        """Evaluate the scaled L1-norm at specific locations.

        Args:
            **variable_locations: Mapping of variable names to locations

        Returns:
            Scaled sum of absolute values (L1-norm)
        """
        value = get_variable_value(self.x, **variable_locations)

        return self.scaling * torch.sum(torch.abs(value))

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
        threshold = self.scaling * prox_scaling
        return torch.nn.functional.relu(location - threshold) - torch.nn.functional.relu(
            -location - threshold)

    def is_subsamplable(self) -> bool:
        """Returns False because L1-norm is not subsamplable."""
        return False

    def subsample(self, indices) -> "L1Norm":
        """Raises NotImplementedError because L1-norm cannot be subsampled."""
        raise NotImplementedError("L1-norm cannot be subsampled.")

    def to_cvxpy(self) -> cp.Expression:
        """Converts the scaled L1-norm to a CVXPY expression.

        This method implicitly assumes that `x` is a Variable.

        Returns:
            CVXPY expression for the scaled L1-norm
        """
        scaling_value = float(self.scaling.item())
        return scaling_value * cp.norm(self.x.to_cvxpy(), 1)
