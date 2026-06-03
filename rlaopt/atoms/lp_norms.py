"""Lp-norm regularizer atoms."""

from abc import ABC, abstractmethod

import torch

from rlaopt.atoms.lp_norm_helpers import (
    project_onto_l1_ball,
    project_onto_l2_ball,
    project_onto_linf_ball,
)
from rlaopt.atoms.nonsmooth_regularizer import NonsmoothRegularizer
from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict


class _Norm(NonsmoothRegularizer, ABC):
    """Base class for scaled Lp-norm regularizers.

    Computes ``scaling * ||x||_p``. Subclasses bind the norm via ``_norm``
    and the projection onto the dual-norm ball via ``_dual_projection``.
    """

    def __init__(self, x: Expression, scaling: float | torch.Tensor = 1.0):
        """Initialize the Lp-norm regularizer with optional scaling."""
        super().__init__(x, scaling_params={"scaling": scaling})

    def forward(self) -> torch.Tensor:
        """Evaluate the scaled Lp-norm."""
        value = self.get_input("x").forward()
        return self.get_buffer("scaling") * self._norm(value)

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Compute the proximal operator via Moreau decomposition."""
        lam = self.get_buffer("scaling") * prox_scaling
        return relevant_variable_values.apply(
            lambda x: x - self._dual_projection(x, lam)
        )

    @abstractmethod
    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the norm used by this regularizer."""

    @abstractmethod
    def _dual_projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Project ``x`` onto the dual-norm ball of the given radius."""


class L1Norm(_Norm):
    """L1-norm regularization atom.

    Computes the scaled L1-norm: scaling * ||x||₁ = scaling * Σᵢ |xᵢ|

    Args:
        x: Expression to apply the L1-norm to.
        scaling: Scaling factor for the L1-norm (default: 1.0).

    Examples:
        >>> x = Variable((100,), name='weights')
        >>> l1 = L1Norm(x, scaling=0.01)
        >>> penalty = l1.forward()
    """

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(value))

    def _dual_projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return project_onto_linf_ball(x, radius)


class L2Norm(_Norm):
    """L2-norm regularization atom.

    Computes the scaled L2-norm: scaling * ||x||₂ = scaling * sqrt(Σᵢ xᵢ²)

    Args:
        x: Expression to apply the L2-norm to.
        scaling: Scaling factor for the L2-norm (default: 1.0).

    Examples:
        >>> x = Variable((100,), name='weights')
        >>> l2 = L2Norm(x, scaling=0.01)
        >>> penalty = l2.forward()
    """

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(value**2))

    def _dual_projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return project_onto_l2_ball(x, radius)


class LInfNorm(_Norm):
    """L-infinity-norm regularization atom.

    Computes the scaled L-infinity norm: scaling * ||x||_∞ = scaling * maxᵢ |xᵢ|

    Args:
        x: Expression to apply the L-infinity norm to.
        scaling: Scaling factor for the L-infinity norm (default: 1.0).

    Examples:
        >>> x = Variable((100,), name='weights')
        >>> linf = LInfNorm(x, scaling=0.01)
        >>> penalty = linf.forward()
    """

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(value))

    def _dual_projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return project_onto_l1_ball(x, radius)
