"""Implementation of the sum squared atom."""

from __future__ import annotations

import torch

from rlaopt.atoms.affine import Affine
from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Expression, Variable


class SumSquares(AtomExpression):
    """Sum of squared elements atom."""

    def __init__(self, x: Variable | Expression):
        """Initializes the sum squared atom.

        Args:
            x: Variable or Expression to apply the sum of squares to.
        """
        super().__init__()

        # Register the input as a Parameter if its a Variable
        # or Module if it's an Expression
        self.register_input(x)

    def is_smooth(self) -> bool:
        """Returns True depending on the smoothness of the expression."""
        input_ = self.get_input()
        if isinstance(input_, Expression):
            return input_.is_smooth()
        else:
            return True

    def forward(self) -> torch.Tensor:
        """Forward pass to compute the sum of squares."""
        value = self.get_input().forward()
        return torch.sum(value**2)

    def is_proxable(self) -> bool:
        """Returns True if the input is a Variable or Affine with Variable root."""
        input_ = self.get_input()
        if isinstance(input_, Variable):
            return True
        elif isinstance(input_, Affine):
            if isinstance(input_.get_input(), Variable):
                return True
        return False

    def prox(self, location, prox_scaling) -> torch.Tensor:
        """Proximal operator for the sum of squares.

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Scaling factor for the proximal operator

        Returns:
            Result of the proximal operator
        """
        input_ = self.get_input()

        if isinstance(input_, Variable):
            return 1 / (1 + 2 * prox_scaling) * location

        elif isinstance(input_, Expression):
            # For expressions, SumSquares is only proxable when
            # input is Affine with a Variable root.

            if isinstance(input_, Affine):
                if isinstance(input_.get_input(), Variable):
                    return _sum_squares_affine_prox(input_, location, prox_scaling)

                else:
                    raise NotImplementedError(
                        "Proximal operator for composition of SumSquares and Affine "
                        "where Affine has non-Variable root is not supported."
                    )

            # SumSquares of general Expression is not proxable
            raise NotImplementedError(
                "Proximal operator for SumSquares and general Expression "
                "is not supported."
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


def _sum_squares_affine_prox(
    input_: Affine, location: torch.Tensor, prox_scaling: float
) -> torch.Tensor:
    """Computes prox operator for ||A @ x + b||^2."""
    # Get data for forming Abar
    n = input_.A.shape[1]
    dtype = input_.A.dtype
    device = input_.A.device

    # Proximal operator is least-squares problem
    # Form Abar = [A; 1/sqrt(2*prox_scaling)*I]
    mu = 1 / (2 * prox_scaling) ** (0.5)
    Abar = torch.cat((input_.A, mu * torch.eye(n, dtype=dtype, device=device)), dim=0)

    # Get QR factorization and setup right handside
    _, R = torch.linalg.qr(Abar, mode="reduced")
    rhs = 1 / (2 * prox_scaling) * location - input_.A.T @ input_.b

    # Find solution via triangular solves
    rhs = torch.linalg.solve_triangular(R.T, rhs.reshape(rhs.shape[0], 1), upper=False)
    return torch.linalg.solve_triangular(R, rhs, upper=True).reshape(
        rhs.shape[0],
    )
