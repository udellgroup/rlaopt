"""Implementation of the sum squared atom."""

import torch

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


class SumSquares(Atom):
    """Sum of squared elements atom."""

    def __init__(self, x: Expression):
        """Initializes the sum squared atom.

        Args:
            x: Expression to apply the sum of squares to.
        """
        super().__init__(exprs={"x": x}, buffers={})

    def is_smooth(self) -> bool:
        """Returns True depending on the smoothness of the expression."""
        return self.get_input("x").is_smooth()

    def forward(self) -> torch.Tensor:
        """Forward pass to compute the sum of squares."""
        value = self.get_input("x").forward()
        return torch.sum(value**2)

    def is_proxable(self) -> bool:
        """Returns True if the input is a Variable."""
        input_ = self.get_input("x")
        if isinstance(input_, Variable):
            return True
        return False

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Proximal operator for the sum of squares."""
        return relevant_variable_values.apply(lambda x: 1 / (1 + 2 * prox_scaling) * x)
