"""Implementation of the sum squared atom."""

from __future__ import annotations

import cvxpy as cp
import torch

from rlaopt.atoms.affine import Affine
from rlaopt.atoms.atom_expression import AtomExpression, InputType
from rlaopt.expression import Expression, Variable


class SumSquares(AtomExpression):
    """Sum of squared elements atom."""

    def __init__(self, x: Variable | Expression):
        """Initializes the sum squared atom.

        Args:
            x: Variable or Expression to apply the sum of squares to.
        """
        super().__init__()

        if not isinstance(x, (Variable, Expression)):
            raise TypeError(f"Expected Variable or Expression, got {type(x)}")

        # Register the input as a parameter if its a variable
        # or module if it's an Expression
        self.register_input(x)

    def is_smooth(self) -> bool:
        """Returns True depending on the smoothness of the expression."""
        if self.input_type == InputType.EXPRESSION:
            return self.get_submodule(self.expr_name).is_smooth()
        else:
            return True

    def forward(self) -> torch.Tensor:
        """Forward pass to compute the sum of squares."""
        if self.input_type == InputType.EXPRESSION:
            value = self.get_submodule(self.expr_name).forward()
        else:
            value = self.get_variable(self.var_name)
        return torch.sum(value**2)

    def is_proxable(self) -> bool:
        """Returns True if the input is a Variable or affine."""
        var = self.get_variable(self.var_name)
        if isinstance(var, torch.nn.Parameter) or isinstance(var, Affine):
            return True
        else:
            return False

    def prox(self, location, prox_scaling) -> torch.Tensor:
        """Proximal operator for the sum of squares.

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Scaling factor for the proximal operator

        Returns:
            Result of the proximal operator
        """
        if self.input_type == InputType.VARIABLE:
            return 1 / (1 + prox_scaling) * location
        elif self.input_type == InputType.EXPRESSION:
            # For expressions, we need to handle the proximal operator differently
            # This is a placeholder; actual implementation may vary
            # based on the expression type
            raise NotImplementedError(
                "Proximal operator for Expression not implemented."
            )

    def is_subsamplable(self) -> bool:
        """Returns True if the input is an affine expression."""
        raise NotImplementedError("Should eventually be True for certain cases.")

    def subsample(self) -> SumSquares:
        """Returns a subsampled version of the SumSquares atom.

        Args:
            indices: Indices to subsample

        Returns:
            New SumSquares atom representing the subsampled version

        Raises:
            NotImplementedError: If the atom does not support subsampling.
        """
        raise NotImplementedError("Subsampling not implemented for SumSquares atom.")

    def to_cvxpy(self) -> cp.Expression:
        """Convert the sum of squares to a CVXPY expression."""
        return cp.sum_squares(self.x.to_cvxpy())
