"""Implementation of the L1-norm atom."""

import cvxpy as cp
import torch


class L1Norm(torch.nn.Module):
    """L1-norm atom."""

    def __init__(self, weight: torch.Tensor, scaling: float = 1.0):
        """Initializes the L1-norm atom."""
        super().__init__()
        self.weight = weight
        self.scaling = scaling

    def __mul__(self, other: float) -> "L1Norm":
        """Allows scaling the L1-norm by a scalar."""
        if isinstance(other, float):
            return L1Norm(self.weight, self.scaling * other)
        return NotImplemented

    def __rmul__(self, other: float) -> "L1Norm":
        """Allows scaling the L1-norm by a scalar."""
        return self * other

    def __truediv__(self, other: float) -> "L1Norm":
        """Allows scaling the L1-norm by a scalar division."""
        return self * (1.0 / other)

    def forward(self) -> torch.Tensor:
        """Computes the L1-norm of the weight tensor."""
        return self.scaling * torch.sum(torch.abs(self.weight))

    def is_smooth(self) -> bool:
        """Returns False because L1-norm is not smooth."""
        return False

    def gradient(self):
        """Returns NotImplemented because L1-norm is non-smooth."""
        return NotImplemented

    def is_proxable(self) -> bool:
        """Returns True because L1-norm is proxable."""
        return True

    def prox(self, location: torch.Tensor) -> torch.Tensor:
        """Proximal operator for the L1-norm."""
        return torch.sign(location) * torch.clamp(
            torch.abs(location) - self.scaling, min=0.0
        )

    def is_subsamplable(self) -> bool:
        """Returns False because L1-norm is not subsamplable."""
        return False

    def subsample(self, indices: torch.Tensor):
        """Returns NotImplemented because L1-norm is not subsamplable."""
        return NotImplemented

    def to_cvxpy(self) -> cp.Expression:
        """Converts the L1-norm to a cvxpy expression."""
        return self.scaling * cp.norm(cp.Variable(self.weight.shape), 1)
