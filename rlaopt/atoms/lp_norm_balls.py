"""Norm ball constraint atoms for optimization."""

from abc import ABC, abstractmethod
from numbers import Real

import torch

from rlaopt.atoms.atom import Atom, AtomDecomposition
from rlaopt.atoms.lp_norm_helpers import (
    project_onto_l1_ball,
    project_onto_l2_ball,
    project_onto_linf_ball,
)
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


class _NormBall(Atom, ABC):
    """Base class for norm ball constraint atoms.

    Provides shared logic for atoms that represent the indicator function of
    a norm ball ``{x : ||x|| <= radius}``. Subclasses define the specific norm
    via ``_norm`` and the projection onto the ball via ``_projection``.
    """

    def __init__(self, x: Expression, radius: float | int | torch.Tensor = 1.0):
        """Initialize the norm ball constraint atom."""
        radius = _validate_radius(radius)
        super().__init__(exprs={"x": x}, buffers={"radius": radius})

    def is_smooth(self) -> bool:
        """Norm ball indicators are non-smooth."""
        return False

    def is_proxable(self) -> bool:
        """Check if the proximal operator is computable."""
        return isinstance(self.get_input("x"), Variable)

    def forward(self) -> torch.Tensor:
        """Evaluate the indicator function of the norm ball."""
        value = self.get_input("x").forward()
        radius = self.get_buffer("radius").to(device=value.device, dtype=value.dtype)
        norm = self._norm(value)
        satisfied = (norm <= radius).item()
        return _indicator(satisfied, value.device, value.dtype)

    def decompose(self) -> list[AtomDecomposition] | None:
        """Decompose the constraint if the input is affine."""
        input_expr = self.get_input("x")
        if not input_expr.is_affine():
            return None

        new_var = Variable.like(input_expr)
        radius = self.get_buffer("radius")
        new_atom = type(self)(new_var, radius=radius)
        return [AtomDecomposition(atom=new_atom, affine_expr=input_expr)]

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Project onto the norm ball (prox of the indicator function)."""
        radius = self.get_buffer("radius")
        return relevant_variable_values.apply(lambda x: self._projection(x, radius))

    @abstractmethod
    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the norm used by this norm ball."""

    @abstractmethod
    def _projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Project ``x`` onto this norm ball of the given radius."""


class L1NormBall(_NormBall):
    """L1-norm ball constraint enforcing ||x||_1 <= radius.

    This atom represents the indicator function of the L1-norm ball:
        0 if ||x||_1 <= radius, +inf otherwise.

    Args:
        x: Expression to constrain.
        radius: Non-negative radius of the L1-norm ball (default: 1.0).
    """

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(value))

    def _projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return project_onto_l1_ball(x, radius)


class L2NormBall(_NormBall):
    """L2-norm (Euclidean) ball constraint enforcing ||x||_2 <= radius.

    This atom represents the indicator function of the Euclidean ball:
        0 if ||x||_2 <= radius, +inf otherwise.

    Args:
        x: Expression to constrain.
        radius: Non-negative radius of the L2-norm ball (default: 1.0).
    """

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(value)

    def _projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return project_onto_l2_ball(x, radius)


class LInfNormBall(_NormBall):
    """L-infinity norm ball constraint enforcing ||x||_inf <= radius.

    This atom represents the indicator function of the L-infinity norm ball:
        0 if ||x||_inf <= radius, +inf otherwise.

    Args:
        x: Expression to constrain.
        radius: Non-negative radius of the L-infinity norm ball (default: 1.0).
    """

    def _norm(self, value: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(value))

    def _projection(self, x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return project_onto_linf_ball(x, radius)


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
        f"radius must be float, int, or Tensor, got {type(radius).__name__}"
    )


def _indicator(
    satisfied: bool, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return 0 if satisfied, infinity otherwise."""
    if satisfied:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return torch.tensor(torch.inf, device=device, dtype=dtype)
