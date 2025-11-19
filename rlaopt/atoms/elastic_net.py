"""Elastic net regularization atom."""

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


class ElasticNet(Atom):
    """Elastic net regularization combining L1 and L2 penalties.

    The elastic net penalty is defined as:
        l1_scaling * ||x||₁ + (l2_scaling / 2) * ||x||₂²

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

    def __init__(
        self,
        x: Variable,
        l1_scaling: float | torch.Tensor = 1.0,
        l2_scaling: float | torch.Tensor = 1.0,
    ):
        """Initialize the elastic net atom.

        Args:
            x: Variable to apply the elastic net penalty to.
            l1_scaling: Scaling factor for the L1-norm penalty. Defaults to 1.0.
            l2_scaling: Scaling factor for the L2-norm penalty. Defaults to 1.0.

        Raises:
            TypeError: If x is not a Variable.
        """
        super().__init__(
            exprs={"x": x},
            buffers={"l1_scaling": l1_scaling, "l2_scaling": l2_scaling},
            variable_names=["x"],
        )

    def is_smooth(self) -> bool:
        """Check if the elastic net is smooth.

        Returns:
            bool: Always False, as the L1 component is non-smooth at zero.
        """
        return False

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

    def is_proxable(self) -> bool:
        """Check if the elastic net has a computable proximal operator.

        Returns:
            bool: Always True, as the elastic net proximal operator
                has a closed-form solution (soft-thresholding with scaling).
        """
        return True

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

    def _scale(self, scaling: float) -> Self:
        """Scale the elastic net regularization atom."""
        new_l1 = self.get_buffer("l1_scaling") * scaling
        new_l2 = self.get_buffer("l2_scaling") * scaling
        return ElasticNet(self.get_input("x"), l1_scaling=new_l1, l2_scaling=new_l2)
