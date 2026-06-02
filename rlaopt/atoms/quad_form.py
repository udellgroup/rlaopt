"""Implementations of quadratic form atoms."""

from abc import ABC

import torch

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


class _QuadForm(Atom, ABC):
    """Base class for quadratic form atoms ``f(x) = x^T Q x``.

    The matrix ``Q`` is stored as a buffer; a value of ``None`` denotes the identity,
    i.e., sum of squares ``sum(x**2)``.
    """

    def __init__(self, x: Expression, Q: torch.Tensor | None = None):
        """Initialize the quadratic-form atom.

        Args:
            x: Expression to apply the quadratic form to.
            Q: Matrix defining the quadratic form, or None for the identity.
        """
        super().__init__(exprs={"x": x}, buffers={"Q": Q})

    def is_smooth(self) -> bool:
        """Returns True depending on the smoothness of the expression."""
        return self.get_input("x").is_smooth()

    def forward(self) -> torch.Tensor:
        """Forward pass to compute the quadratic form."""
        value = self.get_input("x").forward()
        Q = self.get_buffer("Q")
        if Q is None:
            return torch.sum(value**2)
        return value @ (Q @ value)


class SumSquares(_QuadForm):
    """Sum of squares atom, ``f(x) = sum(x**2)``."""

    def __init__(self, x: Expression):
        """Initializes the sum of squares atom.

        Args:
            x: Expression to apply the sum of squares to.
        """
        super().__init__(x, Q=None)

    def is_proxable(self) -> bool:
        """Returns True if the input is a Variable."""
        input_ = self.get_input("x")
        if isinstance(input_, Variable):
            return True
        return False

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Proximal operator for sum of squares."""
        return relevant_variable_values.apply(lambda x: 1 / (1 + 2 * prox_scaling) * x)
