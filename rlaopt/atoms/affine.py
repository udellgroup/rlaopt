"""Affine expression atom for optimization."""

from __future__ import annotations

import cvxpy as cp
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Variable


class Affine(AtomExpression):
    """Affine transformation atom: A @ x + b.

    Represents an affine mapping of a variable, commonly used in linear
    constraints, feature transformations, and linear models.

    Args:
        x: Variable to transform.
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

    def __init__(self, x: Variable, A: torch.Tensor, b: torch.Tensor):
        """Initialize the affine transformation atom.

        Args:
            x: Variable to transform.
            A: Transformation matrix.
            b: Bias/offset vector.

        Raises:
            TypeError: If x is not a Variable.
        """
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        self.register_variable(x)
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
        value = self.get_variable(self.var_name)
        return self.A @ value + self.b

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
            bool: Always True, as rows of A and b can be subsampled.
        """
        return True

    def subsample(self, indices: torch.Tensor) -> Affine:
        """Return a subsampled version of the affine transformation.

        Creates a new affine atom with only the rows of A and b specified by indices.

        Args:
            indices: Row indices to subsample.

        Returns:
            Affine: New affine atom with subsampled transformation.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> A = torch.randn(10, 5)
            >>> b = torch.randn(10)
            >>> affine = Affine(x, A, b)
            >>> sub_affine = affine.subsample(torch.tensor([0, 2, 5]))
            >>> sub_affine.A.shape
            torch.Size([3, 5])
        """
        return Affine(getattr(self, self.var_name), self.A[indices], self.b[indices])

    def to_cvxpy(self) -> cp.Expression:
        """Convert to CVXPY expression (not implemented).

        Returns:
            cp.Expression: Not applicable.

        Raises:
            NotImplementedError: CVXPY conversion not yet implemented for Affine.
        """
        raise NotImplementedError("Affine does not yet support CVXPY conversion.")
