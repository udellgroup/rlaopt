"""Halfspace constraint atom for optimization."""

import torch

from rlaopt.atoms.polyhedron import Polyhedron
from rlaopt.expression import Variable


class Halfspace(Polyhedron):
    """Halfspace constraint atom representing a linear inequality.

    A halfspace constraint restricts a variable to satisfy a linear inequality:
    c^T x <= upper, which defines a half-space in the variable space.

    This is a special case of Polyhedra with a single linear inequality
    constraint, but with an efficient closed-form proximal operator (projection
    onto the halfspace).

    Args:
        x: Variable to constrain.
        c: Normal vector defining the halfspace orientation.
        upper: Upper bound for the linear form c^T x.

    Examples:
        >>> # Constraint: c^T x <= 1
        >>> x = Variable((5,), name='x')
        >>> c = torch.randn(5)
        >>> halfspace = Halfspace(x, c=c, upper=torch.tensor(1.0))

        >>> # Non-negativity for first coordinate: x[0] >= 0
        >>> # Rewritten as: -x[0] <= 0, so c = [-1, 0, 0, ...], upper = 0
        >>> c = torch.zeros(5)
        >>> c[0] = -1.0
        >>> nonneg = Halfspace(x, c=c, upper=torch.tensor(0.0))

        >>> # Use proximal operator for projection
        >>> violating_point = torch.randn(5)
        >>> projected = halfspace.prox(violating_point, prox_scaling=1.0)
    """

    def __init__(self, x: Variable, c: torch.Tensor, upper: torch.Tensor):
        """Initialize the halfspace constraint atom.

        Args:
            x: Variable to constrain.
            c: Normal vector defining the halfspace orientation.
            upper: Upper bound for the linear form c^T x.
        """
        super().__init__(x, A=None, b=None, C=c, lower=None, upper=upper)

    def is_proxable(self) -> bool:
        """Check if the halfspace constraint has a computable proximal operator.

        Returns:
            bool: Always True, as halfspace constraints have a closed-form
                proximal operator (projection onto the halfspace).
        """
        return True

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Compute the proximal operator of the halfspace constraint.

        The proximal operator projects onto the halfspace by moving the point
        perpendicular to the boundary until it satisfies c^T x <= upper.
        The prox_scaling parameter is unused because the projection is
        independent of scaling.

        Args:
            location: Point at which to evaluate the proximal operator.
            prox_scaling: Scaling factor (unused for halfspace constraints).

        Returns:
            torch.Tensor: Projected point satisfying the halfspace constraint.
                If the point already satisfies the constraint, it is returned
                unchanged.
        """
        c_norm = torch.linalg.norm(self.C, 2)
        r = torch.dot(self.C, location) - self.upper
        zero = torch.tensor(0.0, device=r.device)
        return location - torch.maximum(r, zero) * self.C / c_norm**2
