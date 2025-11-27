"""Elastic net regularization atom."""

import torch

from rlaopt.atoms.nonsmooth_regularizer import NonsmoothRegularizer
from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict


class ElasticNet(NonsmoothRegularizer):
    """Elastic net regularization combining L1 and L2 penalties.

    The elastic net penalty is defined as:
        l1_scaling * ||x||₁ + (l2_scaling / 2) * ||x||₂²

    Args:
        x: Expression to apply the elastic net penalty to.
        l1_scaling: Scaling factor for the L1-norm penalty. Defaults to 1.0.
        l2_scaling: Scaling factor for the L2-norm penalty. Defaults to 1.0.

    Raises:
        TypeError: If x is not an Expression.

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

    def __init__(
        self,
        x: Expression,
        l1_scaling: float | torch.Tensor = 1.0,
        l2_scaling: float | torch.Tensor = 1.0,
    ):
        """Initialize the elastic net atom.

        Args:
            x: Expression to apply the elastic net penalty to.
            l1_scaling: Scaling factor for the L1-norm penalty. Defaults to 1.0.
            l2_scaling: Scaling factor for the L2-norm penalty. Defaults to 1.0.
        """
        super().__init__(
            x,
            scaling_params={"l1_scaling": l1_scaling, "l2_scaling": l2_scaling},
        )

    def forward(self) -> torch.Tensor:
        """Evaluate the elastic net penalty at the current variable value.

        Returns:
            torch.Tensor: The elastic net penalty value:
                l1_scaling * ||x||₁ + (l2_scaling / 2) * ||x||₂²
        """
        value = self.get_input("x").forward()

        l1_norm = torch.sum(torch.abs(value))
        l2_norm = torch.sum(value**2)

        return (
            self.get_buffer("l1_scaling") * l1_norm
            + (self.get_buffer("l2_scaling") / 2) * l2_norm
        )

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Compute the proximal operator of the elastic net.

        The proximal operator applies soft-thresholding followed by scaling
        to account for the L2 regularization term.
        """
        l2_term = 1 + prox_scaling * self.get_buffer("l2_scaling")
        threshold = self.get_buffer("l1_scaling") * prox_scaling

        def soft_threshold_with_scaling(x: torch.Tensor) -> torch.Tensor:
            return (
                torch.nn.functional.relu(x - threshold)
                - torch.nn.functional.relu(-x - threshold)
            ) / l2_term

        return relevant_variable_values.apply(soft_threshold_with_scaling)
