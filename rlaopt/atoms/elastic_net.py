"""Implementation of the elastic net atom."""

import cvxpy as cp
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable
from rlaopt.atoms._utils import get_variable_value


class ElasticNet(AtomExpression):
    """Elastic net atom."""

    def __init__(self, x: Variable, l1_scaling: float = 1.0, l2_scaling: float = 1.0):
        """Initializes the elastic net atom with optional scaling.

        Args:
            x: Variable to apply the elastic net to.
               Must be an instance of Variable.
            l1_scaling: Scaling factor for the L1-norm (default: 1.0).
                        Can be a float or a torch.Tensor.
            l2_scaling: Scaling factor for the L2-norm (default: 1.0).
                        Can be a float or a torch.Tensor.
        """
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        # Register the variable as a parameter
        self.register_parameter("x", x.value)

        # Register the L1 scaling factor
        if isinstance(l1_scaling, torch.Tensor):
            self.register_buffer("l1_scaling", l1_scaling)
        elif isinstance(l1_scaling, torch.nn.Parameter):
            self.register_buffer("l1_scaling", l1_scaling.data)
        else:
            self.register_buffer("l1_scaling", torch.tensor(float(l1_scaling)))

        # Register the L2 scaling factor
        if isinstance(l2_scaling, torch.Tensor):
            self.register_buffer("l2_scaling", l2_scaling)
        elif isinstance(l2_scaling, torch.nn.Parameter):
            self.register_buffer("l2_scaling", l2_scaling.data)
        else:
            self.register_buffer("l2_scaling", torch.tensor(float(l2_scaling)))

    def is_smooth(self) -> bool:
        """Returns False because elastic net is not smooth."""
        return False

    def evaluate_at(self, **variable_locations):
        """Evaluate the elastic net at specific locations.

        Args:
            **variable_locations: Mapping of variable names to locations

        Returns:
            Elastic net value
        """
        value = get_variable_value(self.x, **variable_locations)

        l1_norm = torch.sum(torch.abs(value))
        l2_norm = torch.sum(value**2)

        return self.l1_scaling * l1_norm + (self.l2_scaling / 2) * l2_norm

    def is_proxable(self) -> bool:
        """Returns True because elastic net is proxable."""
        return True

    def prox(self, location, prox_scaling) -> torch.Tensor:
        """Proximal operator for the elastic net.

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Scaling factor for the proximal operator
        Returns:
            Result of the proximal operator
        """
        l2_term = 1 + prox_scaling * self.l2_scaling
        threshold = self.l1_scaling * prox_scaling
        return (
            torch.nn.functional.relu(location - threshold)
            - torch.nn.functional.relu(-location - threshold)
        ) / l2_term
