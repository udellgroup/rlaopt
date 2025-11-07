"""Affine expression atom for optimization."""

import torch

from typing_extension import Self

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Expression, Variable

import linops as lo


class Linop(AtomExpression):
    """Linear operator atom: A @ x.

    Represents an affine mapping of a variable or output of an expression.

    Args:
        x: Variable or Expression to transform.
        A: Transformation linop.

    Raises:
        TypeError: If x is not a Variable or Expression.

    Examples:
        >>> x = Variable((5,), name='x')
        >>> A = torch.randn(3, 5)
        >>> b = torch.randn(3)
        >>> affine = Affine(x, A, b)
        >>> result = affine.forward()  # Computes A @ x + b
    """

    def __init__(self, A: lo.LinearOperator, x: Variable | Expression):
        """Initialize the affine transformation atom.

        Args:
            x: Variable or Affine Expression to transform.
            A: Transformation operator.

        Raises:
            TypeError: If x is not a Variable or Expression.
        """
        super().__init__()

        self.register_input(x)
        self.op = A

    def is_smooth(self) -> bool:
        """Check if the affine transformation is smooth.

        Returns:
            bool: Always True, as affine functions are smooth everywhere.
        """
        return True

    def forward(self) -> torch.Tensor:
        """Evaluate the affine transformation at the current variable value.

        Returns:
            torch.Tensor: Result of A @ x + b.
        """
        input_ = self.get_input().forward()
        return self.op @ input_

    def is_proxable(self) -> bool:
        """Check if the affine transformation has a computable proximal operator.

        Returns:
            bool: Always False, as affine functions are not proxable.
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
        """Check if the affine transformation supports subsampling.

        Returns:
            bool: Always False, we do not support subsampling currently.
        """
        return True

    def subsample(self, indices: torch.Tensor) -> Self:
        """Return a subsampled version of the affine transformation.

        Returns:
            Affine: Not applicable

        Raises:
            NotImplementedError: Affine transformations are not proxable.
        """
        return Linop( self.op[indices], self.get_input())
