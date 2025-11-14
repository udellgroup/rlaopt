"""Affine expression atom for optimization."""

import torch

from typing_extensions import Self

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Expression, Variable

import linops as lo


class Linop(AtomExpression):
    """Linear operator atom: A @ x.

    Represents a linear mapping of a variable or output of an expression.

    Args:
        x: Variable or Expression to transform.
        A: Transformation linop.

    Raises:
        TypeError: If x is not a Variable or Expression.

    Examples:
        >>> x = Variable((5,), name='x')
        >>> A = torch.randn(3, 5)
        >>> b = torch.randn(3)
        >>> linop = Linop(x, A, b)
        >>> result = linop.forward()  # Computes A @ x + b
    """

    def __init__(self, A: lo.LinearOperator, x: Variable | Expression):
        """Initialize the linop transformation atom.

        Args:
            x: Variable or Expression to transform.
            A: Transformation operator.

        Raises:
            TypeError: If x is not a Variable or Expression.
        """
        super().__init__()

        self.register_input(x)
        self.op = A

    def is_smooth(self) -> bool:
        """Check if the linear transformation is smooth.

        Returns:
            bool: Always True, as linear functions are smooth everywhere.
        """
        return True

    def forward(self) -> torch.Tensor:
        """Evaluate the linear transformation at the current variable value.

        Returns:
            torch.Tensor: Result of A @ x + b.
        """
        input_ = self.get_input().forward()
        return self.op @ input_

    def is_proxable(self) -> bool:
        """Check if the linear transformation has a computable proximal operator.

        Returns:
            bool: Always False, as linear functions are not proxable.
        """
        return False

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Compute the proximal operator (not supported).

        Args:
            location: Point at which to evaluate the proximal operator (unused).
            prox_scaling: Scaling factor for the proximal operator (unused).

        Returns:
            torch.Tensor: Not applicable.

        Raises:
            NotImplementedError: Affine transformations are not proxable.
        """
        raise NotImplementedError("Affine is not proxable.")

    def is_subsamplable(self) -> bool:
        """Check if the linear transformation supports subsampling.

        Returns:
            bool: Always True.
        """
        return True

    def subsample(self, indices: torch.Tensor) -> Self:
        """Return a subsampled version of the linear transformation.

        Returns:
            Linop: subsampled linop
        """
        return Linop( self.op[indices], self.get_input())
