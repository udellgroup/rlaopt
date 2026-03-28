"""L-infinity norm ball constraint atom for optimization."""

from numbers import Real

import torch

from rlaopt.atoms.box import Box
from rlaopt.expression import Variable


class LInfNormBall(Box):
    """L-infinity norm ball constraint enforcing ||x||_inf <= radius.

    This atom represents the indicator function of the L-infinity norm ball:
        0 if ||x||_inf <= radius, +inf otherwise.

    Args:
        x: Variable to constrain.
        radius: Non-negative radius of the L-infinity norm ball (default: 1.0).
    """

    def __init__(
        self, 
        x: Variable, 
        radius: float | int | torch.Tensor = 1.0
    ):
        """Initialize the L-infinity norm ball constraint atom."""
        radius = _validate_radius(radius)
        if torch.is_tensor(radius):
            upper = radius
        else:
            upper = float(radius)
        lower = -upper
        super().__init__(x, lower=lower, upper=upper)


def _validate_radius(radius: float | int | torch.Tensor) -> float | torch.Tensor:
    """Validate and normalize the radius parameter."""
    if isinstance(radius, Real):
        if radius < 0:
            raise ValueError("radius must be non-negative")
        return float(radius)
    if torch.is_tensor(radius):
        if radius.numel() != 1:
            raise ValueError("radius must be a scalar tensor")
        if torch.any(radius < 0):
            raise ValueError("radius must be non-negative")
        return radius
    raise TypeError(
        f"radius must be float, int, or Tensor, got {type(radius).__name__}")
