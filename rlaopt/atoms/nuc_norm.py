"""Nuclear norm atom for matrix regularization."""

from __future__ import annotations

import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Variable


class NucNorm(AtomExpression):
    """Nuclear norm (sum of singular values) of a matrix variable.

    The nuclear norm is defined as the sum of the singular values of a matrix.
    It is commonly used as a convex relaxation of the rank function in low-rank
    matrix optimization problems.

    The atom computes: scaling * ||X||_* = scaling * Σᵢ σᵢ(X)
    where σᵢ(X) are the singular values of X.

    Args:
        x: 2D matrix variable to apply the nuclear norm to.
        scaling: Scaling factor for the nuclear norm. Defaults to 1.0.

    Raises:
        TypeError: If x is not a Variable.
        ValueError: If x is not a 2D matrix.

    Examples:
        >>> X = Variable((10, 5), name='X')
        >>> nuc_norm = NucNorm(X, scaling=0.1)
        >>> loss = nuc_norm.forward()
    """

    def __init__(
        self, x: Variable, scaling: float | torch.Tensor | torch.nn.Parameter = 1.0
    ):
        """Initialize the nuclear norm atom.

        Args:
            x: 2D matrix variable to apply the nuclear norm to.
            scaling: Scaling factor for the nuclear norm. Defaults to 1.0.

        Raises:
            TypeError: If x is not a Variable.
            ValueError: If x is not a 2D matrix.
        """
        super().__init__()
        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")
        if x.value.data.dim() != 2:
            raise ValueError(
                f"Variable value must be 2D Tensor, "
                f"but got {x.value.data.dim()}D Tensor."
            )
        # Register the variable as a parameter
        self.register_variable(x)
        # Register the scaling factor
        self.register_atom_buffer("scaling", scaling)

    def is_smooth(self) -> bool:
        """Check if the nuclear norm is smooth.

        Returns:
            bool: Always False, as the nuclear norm is non-smooth.
        """
        return False

    def forward(self) -> torch.Tensor:
        """Evaluate the nuclear norm at the registered variable value.

        Returns:
            torch.Tensor: The scaled sum of singular values.
        """
        value = self.get_variable(self.var_name)
        S = torch.linalg.svdvals(value)
        return self.scaling * torch.sum(S)

    def is_proxable(self) -> bool:
        """Check if the nuclear norm has a computable proximal operator.

        Returns:
            bool: Always True, as the nuclear norm is proxable.
        """
        return True

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Compute the proximal operator of the nuclear norm.

        The proximal operator performs singular value soft-thresholding.

        Args:
            location: Point at which to evaluate the proximal operator.
            prox_scaling: Scaling factor for the proximal operator.

        Returns:
            torch.Tensor: Result of the proximal operator (soft-thresholded matrix).
        """
        U, S, Vt = torch.linalg.svd(location, full_matrices=False)
        S = torch.nn.functional.relu(S - prox_scaling * self.scaling)
        return (U * S) @ Vt

    def is_subsamplable(self) -> bool:
        """Check if the nuclear norm supports subsampling.

        Returns:
            bool: Always False, as nuclear norm cannot be subsampled.
        """
        return False

    def subsample(self, indices: torch.Tensor) -> NucNorm:
        """Subsample the nuclear norm (not supported).

        Args:
            indices: Indices to subsample (unused).

        Returns:
            NucNorm: Not applicable.

        Raises:
            NotImplementedError: Nuclear norm cannot be subsampled.
        """
        raise NotImplementedError("Nuclear norm cannot be subsampled")

    def to_cvxpy(self):
        """Convert to CVXPY expression (not supported).

        Raises:
            NotImplementedError: CVXPY conversion not implemented for nuclear norm.
        """
        raise NotImplementedError("NucNorm does not support conversion to cvxpy")
