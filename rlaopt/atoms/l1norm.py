"""Implementation of the L1-norm atom."""

import cvxpy as cp
import torch

from rlaopt.atoms.atom import Atom


class L1Norm(Atom):
    """L1-norm atom."""

    def __init__(self, scaling: float = 1.0):
        """Initializes the L1-norm atom."""
        super().__init__(scaling=scaling)

    def _forward_impl(self, location: torch.Tensor) -> torch.Tensor:
        """Unscaled evaluation of the L1-norm."""
        return torch.sum(torch.abs(location))

    def is_smooth(self) -> bool:
        """Returns False because L1-norm is not smooth."""
        return False

    def is_proxable(self) -> bool:
        """Returns True because L1-norm is proxable."""
        return True

    def prox(self, location: torch.Tensor) -> torch.Tensor:
        """Proximal operator for the L1-norm.

        The proximal operator is computed by soft-thresholding the input location.
        """
        return torch.sign(location) * torch.clamp(
            torch.abs(location) - self.scaling, min=0.0
        )

    def is_subsamplable(self) -> bool:
        """Returns False because L1-norm is not subsamplable."""
        return False

    def subsample(self, indices: torch.Tensor):
        """Raises NotImplementedError because L1-norm cannot be subsampled."""
        raise NotImplementedError("L1-norm cannot be subsampled.")

    def to_cvxpy(self, variable_or_expr: cp.Variable | cp.Expression) -> cp.Expression:
        """Converts the L1-norm to a cvxpy expression."""
        return self.scaling * cp.norm(variable_or_expr, 1)

    def __mul__(self, scalar: float) -> "L1Norm":
        """Allows scaling the L1-norm by a scalar."""
        if isinstance(scalar, float):
            return L1Norm(self.scaling * scalar)
        return NotImplemented
