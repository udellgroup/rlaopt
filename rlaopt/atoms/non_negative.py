"""Non-negativity constraint atom for optimization."""

from rlaopt.atoms.box import Box
from rlaopt.expression import Variable


class NonNegative(Box):
    """Non-negativity constraint atom enforcing x >= 0.

    A convenience class for box constraints with lower bound of zero and
    no upper bound, enforcing that all elements of the variable are
    non-negative.

    This is equivalent to Box(x, lower=0, upper=None) but provides a more
    semantic interface for this common constraint.

    Args:
        x: Variable to constrain to be non-negative.

    Examples:
        >>> # Constrain variable to be non-negative
        >>> x = Variable((10,), name='x')
        >>> nonneg = NonNegative(x)

        >>> # Use proximal operator for projection
        >>> point_with_negatives = torch.randn(10)
        >>> projected = nonneg.prox(point_with_negatives, prox_scaling=1.0)
        >>> # All negative values are clamped to 0
    """

    def __init__(self, x: Variable):
        """Initialize the non-negativity constraint atom.

        Args:
            x: Variable to constrain to be non-negative.
        """
        super().__init__(x, lower=0.0, upper=None)
