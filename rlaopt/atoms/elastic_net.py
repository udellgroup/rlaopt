"""Elastic net regularization atom."""

import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression import Variable


class ElasticNet(AtomExpression):
    """Elastic net regularization combining L1 and L2 penalties.

    The elastic net penalty is defined as:
        l1_scaling * ||x||₁ + (l2_scaling / 2) * ||x||₂²

    This combines the sparsity-inducing property of L1 regularization with
    the grouping effect of L2 regularization, making it particularly useful
    when dealing with correlated features.

    Args:
        x: Variable to apply the elastic net penalty to.
        l1_scaling: Scaling factor for the L1-norm penalty. Defaults to 1.0.
        l2_scaling: Scaling factor for the L2-norm penalty. Defaults to 1.0.

    Raises:
        TypeError: If x is not a Variable.

    Examples:
        >>> x = Variable((100,), name='weights')
        >>> # Standard elastic net with equal L1 and L2 contribution
        >>> elastic = ElasticNet(x, l1_scaling=0.5, l2_scaling=0.5)
        >>> penalty = elastic.forward()

        >>> # Lasso-like (emphasize sparsity)
        >>> elastic_lasso = ElasticNet(x, l1_scaling=1.0, l2_scaling=0.1)

        >>> # Ridge-like (emphasize smoothness)
        >>> elastic_ridge = ElasticNet(x, l1_scaling=0.1, l2_scaling=1.0)
    """

    def __init__(self, x: Variable, l1_scaling: float = 1.0, l2_scaling: float = 1.0):
        """Initialize the elastic net atom.

        Args:
            x: Variable to apply the elastic net penalty to.
            l1_scaling: Scaling factor for the L1-norm penalty. Defaults to 1.0.
            l2_scaling: Scaling factor for the L2-norm penalty. Defaults to 1.0.

        Raises:
            TypeError: If x is not a Variable.
        """
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        # Register the input variable as a parameter
        self.register_variable(x)

        # Register the L1 and L2 scaling factors as buffers
        self.register_atom_buffer("l1_scaling", l1_scaling)
        self.register_atom_buffer("l2_scaling", l2_scaling)

    def is_smooth(self) -> bool:
        """Check if the elastic net is smooth.

        Returns:
            bool: Always False, as the L1 component is non-smooth at zero.
        """
        return False

    def is_subsamplable(self) -> bool:
        """Check if the elastic net supports subsampling.

        Returns:
            bool: Always False, as regularization operates on all parameters.
        """
        return False

    def subsample(self, indices: torch.Tensor) -> "ElasticNet":
        """Subsample the elastic net (not supported).

        Args:
            indices: Indices to subsample (unused).

        Returns:
            ElasticNet: Not applicable.

        Raises:
            NotImplementedError: Elastic net does not support subsampling.
        """
        raise NotImplementedError("Elastic net atom does not support subsampling")

    def to_cvxpy(self):
        """Convert to CVXPY expression (not supported).

        Raises:
            NotImplementedError: CVXPY conversion not implemented for elastic net.
        """
        raise NotImplementedError("Conversion to cvxpy is not supported.")

    def forward(self) -> torch.Tensor:
        """Evaluate the elastic net penalty at the current variable value.

        Returns:
            torch.Tensor: The elastic net penalty value:
                l1_scaling * ||x||₁ + (l2_scaling / 2) * ||x||₂²
        """
        value = self.get_variable(self.var_name)

        l1_norm = torch.sum(torch.abs(value))
        l2_norm = torch.sum(value**2)

        return self.l1_scaling * l1_norm + (self.l2_scaling / 2) * l2_norm

    def is_proxable(self) -> bool:
        """Check if the elastic net has a computable proximal operator.

        Returns:
            bool: Always True, as the elastic net proximal operator
                has a closed-form solution (soft-thresholding with scaling).
        """
        return True

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Compute the proximal operator of the elastic net.

        The proximal operator applies soft-thresholding followed by scaling
        to account for the L2 regularization term.

        Args:
            location: Point at which to evaluate the proximal operator.
            prox_scaling: Scaling factor for the proximal operator.

        Returns:
            torch.Tensor: Result of the proximal operator.
                Formula: soft_threshold(location, λ₁ * ρ) / (1 + ρ * λ₂)
                where λ₁ = l1_scaling, λ₂ = l2_scaling, ρ = prox_scaling.
        """
        l2_term = 1 + prox_scaling * self.l2_scaling
        threshold = self.l1_scaling * prox_scaling
        return (
            torch.nn.functional.relu(location - threshold)
            - torch.nn.functional.relu(-location - threshold)
        ) / l2_term
