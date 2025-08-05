"""Implementation of the sum squared atom."""

import cvxpy as cp
import torch

from rlaopt.atoms.atom import Atom
from rlaopt.atoms.affine_atom import AffineAtom
from rlaopt.expression.expression import Expression
from ..expression.variable import Variable


class SumSquares(Atom):
    """Sum of squared elements atom."""

    def __init__(self, x: Variable | Expression):
        """Initializes the sum squared atom.

        Args:
            x: Variable or Expression to apply the sum of squares to.
        """
        super().__init__()

        if not isinstance(x, (Variable, Expression)):
            raise TypeError(f"Expected Variable or Expression, got {type(x)}")

        # Register the variable as a parameter or module if it's an Expression
        if isinstance(x, Expression):
            self.add_module("x", x)
        elif isinstance(x, Variable):
            self.register_parameter("x", x.value)

    def is_smooth(self) -> bool:
        """Returns True depending on the smoothness of the expression."""
        if isinstance(self.x, Expression):
            return self.x.is_smooth()
        elif isinstance(self.x, Variable):
            return True

    def evaluate_at(self, **variable_locations):
        """Evaluate the sum of squares at specific locations.

        Args:
            **variable_locations: Mapping of variable names to locations

        Returns:
            Sum of squares
        """

        # If x is an Expression, pass the variable_locations to it
        if isinstance(self.x, Expression):
            value = self.x.evaluate_at(**variable_locations)

        # Otherwise use registered value    
        else:
            value = self.x

        return torch.sum(value ** 2)

    def is_proxable(self) -> bool:
        """Returns True if the input is a Variable or affine."""
        if isinstance(self.x, Variable) or isinstance(self.x, AffineAtom):
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
        if isinstance(self.x, Variable):
            return 1 / (1 + prox_scaling) * location
        elif isinstance(self.x, Expression):
            # For expressions, we need to handle the proximal operator differently
            # This is a placeholder; actual implementation may vary
            # based on the expression type
             raise NotImplementedError(
                 "Proximal operator for Expression not implemented."
            )
            
    def is_subsamplable(self) -> bool:
        """Returns True if the input is a an affine expression."""
        raise NotImplementedError("Should eventually be True for certain cases.")

    def subsample(self) -> "SumSquares":
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
