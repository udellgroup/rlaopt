"""Affine expression atom for optimization."""

from __future__ import annotations

import cvxpy as cp
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Expression, Variable


class Affine(AtomExpression):
    """Affine transformation atom: A @ x + b.

    Represents an affine mapping of a variable or output of an expression.

    Args:
        x: Variable or Expression to transform.
        A: Transformation matrix.
        b: Bias/offset vector.

    Raises:
        TypeError: If x is not a Variable.

    Examples:
        >>> x = Variable((5,), name='x')
        >>> A = torch.randn(3, 5)
        >>> b = torch.randn(3)
        >>> affine = Affine(x, A, b)
        >>> result = affine.forward()  # Computes A @ x + b
    """

    def __init__(self, x: Variable | Expression, A: torch.Tensor, b: torch.Tensor):
        """Initialize the affine transformation atom.

        Args:
            x: Variable or Affine Expression to transform.
            A: Transformation matrix.
            b: Bias/offset vector.

        Raises:
            TypeError: If x is not a Variable or Expression.
        """
        super().__init__()

        self.register_input(x)
        self.register_atom_buffer("A", A)
        self.register_atom_buffer("b", b)

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
        input_ = self.get_input()
        if isinstance(input_, torch.nn.Parameter):
            return self.A @ input_ + self.b
        elif isinstance(input_, Expression):
            return self.A @ input_.forward() + self.b

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
        return False

    def subsample(self, indices: torch.Tensor) -> Affine:
        """Return a subsampled version of the affine transformation.

        Returns:
            Affine: Not applicable

        Raises:
            NotImplementedError: Affine transformations are not proxable.
        """
        raise NotImplementedError("Affine atom does not support subsampling")

    def to_cvxpy(self) -> cp.Expression:
        """Convert to CVXPY expression (not implemented).

        Returns:
            cp.Expression: Not applicable.

        Raises:
            NotImplementedError: CVXPY conversion not yet implemented for Affine.
        """
        raise NotImplementedError("Affine does not yet support CVXPY conversion.")
