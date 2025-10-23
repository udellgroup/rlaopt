"""Linear equality constraint atom for optimization."""

import torch

from rlaopt.atoms.polyhedron import Polyhedron
from rlaopt.expression import Variable


class LinearEquality(Polyhedron):
    """Linear equality constraint atom enforcing A @ x = b.

    Represents a system of linear equality constraints. Unlike the general
    Polyhedron class, this provides an efficient closed-form proximal operator
    (projection onto the affine subspace) via QR factorization.
    Checks for feasibility at initialization, by determining the rank of A via
    QR factorization.
    If A is not full-row rank, then an error is raised.
    When A is full-row rank, the R factor is cached and used for efficienctly computing
    the projection onto the constraint set.

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
        >>> constraint = LinEqConstraint(x, A, b)

        >>> # Multiple equality constraints
        >>> x = Variable((5,), name='x')
        >>> A = torch.randn(3, 5)
        >>> b = torch.randn(3)
        >>> constraint = LinEqConstraint(x, A, b)

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
        self._R = _factor_and_check_feasibility(self.A)

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
        by solving the constrained least-squares problem.

        Args:
            location: Point at which to evaluate the proximal operator.
            prox_scaling: Scaling factor (unused for equality constraints).

        Returns:
            torch.Tensor: Projection of location onto the affine subspace.
        """
        # NOTE(Zach): We may want to add a check where we just return the location
        # if the residual is sufficiently small.
        r = self.A @ location - self.b
        temp = torch.linalg.solve_triangular(
            self.R,
            torch.linalg.solve_triangular(
                self.R.T, r.reshape(r.shape[0], 1), upper=False
            ),
            upper=True,
        )
        return location - self.A.T @ temp.reshape(
            temp.shape[0],
        )

    @property
    def R(self) -> torch.Tensor:
        """Returns cached R factor from QR factorization of A.T.

        This alloow for efficient computation the projection to the constraint set.

        """
        return self._R


# NOTE(Zach): This helper may be moved or refactored later on.
def _factor_and_check_feasibility(A: torch.Tensor) -> torch.Tensor | None:
    _, R = torch.linalg.qr(A.T, mode="r")

    diag_R = torch.diagonal(R)

    m, n = R.shape

    # Set relative tolerance based on dtype
    if R.dtype == torch.bfloat16:
        rtol = 7.8e-3
    elif R.dtype == torch.float32:
        rtol = 1.2e-7
    elif R.dtype == torch.float64:
        rtol = 2.2e-16
    else:
        raise ValueError(f"Unsupported dtype: {R.dtype}")

    # Scale by matrix size
    rtol = rtol * max(m, n)

    max_diag = torch.max(torch.abs(diag_R))
    tol = rtol * max_diag

    rank = torch.sum(torch.abs(diag_R) > tol).item()

    if rank == n:  # Should equal number of columns
        return R
    else:
        raise ValueError(
            "The provided constraint matrix A is rank deficient, and so does not "
            "define a valid equality constraint"
        )
