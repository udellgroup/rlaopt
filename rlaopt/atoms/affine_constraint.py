"""Affine equality constraint atom for optimization."""

import torch

from rlaopt.atoms.polyhedra import Polyhedra
from rlaopt.expression import Variable


class AffineConstraint(Polyhedra):
    """Affine equality constraint atom enforcing A @ x = b.

    Represents a system of linear equality constraints. Unlike the general
    Polyhedra class, this provides an efficient closed-form proximal operator
    (projection onto the affine subspace) via Cholesky factorization.

    The projection solves: argmin_z ||z - location||² subject to A @ z = b

    Args:
        x: Variable to constrain.
        A: Constraint matrix defining the linear system.
        b: Right-hand side vector of the equality constraints.

    Examples:
        >>> # Single equality constraint: x[0] + x[1] = 1
        >>> x = Variable((2,), name='x')
        >>> A = torch.tensor([[1.0, 1.0]])
        >>> b = torch.tensor([1.0])
        >>> constraint = AffineConstraint(x, A, b)

        >>> # Multiple equality constraints
        >>> x = Variable((5,), name='x')
        >>> A = torch.randn(3, 5)
        >>> b = torch.randn(3)
        >>> constraint = AffineConstraint(x, A, b)

        >>> # Use proximal operator for projection onto affine subspace
        >>> unconstrained_point = torch.randn(5)
        >>> projected = constraint.prox(unconstrained_point, prox_scaling=1.0)
        >>> # Verify: A @ projected should equal b
    """

    def __init__(self, x: Variable, A: torch.Tensor, b: torch.Tensor):
        """Initialize the affine equality constraint atom.

        Args:
            x: Variable to constrain.
            A: Constraint matrix defining the linear system.
            b: Right-hand side vector of the equality constraints.
        """
        super().__init__(x, A=A, b=b, C=None, lower=None, upper=None)
        self._prox = _build_prox(A, b, prox_mode="exact")

    def is_proxable(self) -> bool:
        """Check if the constraint has a computable proximal operator.

        Returns:
            bool: Always True, as affine equality constraints have a
                closed-form proximal operator (projection via Cholesky).
        """
        return True

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Compute the proximal operator of the affine constraint.

        Projects the given location onto the affine subspace {x : A @ x = b}
        by solving the constrained least-squares problem. The prox_scaling
        parameter is unused because the projection is independent of scaling.

        Args:
            location: Point at which to evaluate the proximal operator.
            prox_scaling: Scaling factor (unused for equality constraints).

        Returns:
            torch.Tensor: Projection of location onto the affine subspace.
        """
        return self._prox(location, prox_scaling)


def _build_prox(A: torch.Tensor, b: torch.Tensor, prox_mode: str):
    """Build the proximal operator function for affine constraints.

    Constructs a function that projects points onto the affine subspace
    defined by A @ x = b using the specified method.

    Args:
        A: Constraint matrix.
        b: Right-hand side vector.
        prox_mode: Method for computing the projection. Currently only
            "exact" is supported, which uses Cholesky factorization.

    Returns:
        Callable: Function that computes the proximal operator.

    Notes:
        The exact method solves: x* = location - A^T (A A^T)^(-1) (A @ location - b)
        using Cholesky factorization of the Gram matrix G = A A^T for efficiency.
    """
    if prox_mode == "exact":

        def prox(location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
            """Compute exact projection via Cholesky factorization.

            Args:
                location: Point to project.
                prox_scaling: Unused (projection is scale-independent).

            Returns:
                torch.Tensor: Projected point on the affine subspace.
            """
            r = A @ location - b
            G = A @ A.T
            L = torch.linalg.cholesky(G)

            # Solve G^(-1) @ r via forward and backward substitution
            temp = torch.linalg.solve_triangular(
                L.T,
                torch.linalg.solve_triangular(L, r.reshape(r.shape[0], 1), upper=False),
                upper=True,
            )
            return location - A.T @ temp.reshape(
                temp.shape[0],
            )

    return prox
