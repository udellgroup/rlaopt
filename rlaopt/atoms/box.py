"""Box constraint atom for optimization."""

import torch

from rlaopt.atoms.polyhedron import Polyhedron
from rlaopt.expression import Variable


class Box(Polyhedron):
    """Box constraint atom representing elementwise bounds.

    A box constraint restricts each element of a variable to lie within
    specified lower and upper bounds: lower <= x <= upper.

    This is a special case of Polyhedra with identity inequality constraints
    (C = I), but with an efficient closed-form proximal operator (projection
    via clamping).

    Args:
        x: Variable to constrain.
        lower: Lower bound vector (optional). If None, defaults to -infinity.
        upper: Upper bound vector (optional). If None, defaults to +infinity.

    Examples:
        >>> # Standard box constraint: 0 <= x <= 1
        >>> x = Variable((10,), name='x')
        >>> box = Box(x, lower=torch.zeros(10), upper=torch.ones(10))

        >>> # One-sided constraint: x >= 0 (non-negativity)
        >>> box_nonneg = Box(x, lower=torch.zeros(10))

        >>> # One-sided constraint: x <= 1
        >>> box_upper = Box(x, upper=torch.ones(10))

        >>> # Use proximal operator for projection
        >>> out_of_bounds = torch.randn(10)
        >>> projected = box.prox(out_of_bounds, prox_scaling=1.0)
    """

    def __init__(
        self,
        x: Variable,
        lower: torch.Tensor = None,
        upper: torch.Tensor = None,
    ):
        """Initialize the box constraint atom.

        Args:
            x: Variable to constrain.
            lower: Lower bound vector (optional).
            upper: Upper bound vector (optional).
        """
        super().__init__(x, A=None, b=None, C=None, lower=lower, upper=upper)

    def is_proxable(self) -> bool:
        """Check if the box constraint has a computable proximal operator.

        Returns:
            bool: Always True, as box constraints have a closed-form
                proximal operator (projection via clamping).
        """
        return True

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Compute the proximal operator of the box constraint.

        The proximal operator projects onto the box by clamping each element
        to lie within [lower, upper]. The prox_scaling parameter is unused
        because the projection is independent of scaling.

        Args:
            location: Point at which to evaluate the proximal operator.
            prox_scaling: Scaling factor (unused for box constraints).

        Returns:
            torch.Tensor: Projected point with each element clamped to
                [lower, upper].
        """
        return torch.clamp(location, self.lower, self.upper)
