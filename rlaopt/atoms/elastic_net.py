"""Elastic net regularization atom."""

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom, AtomDecomposition
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


class ElasticNet(Atom):
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

        Raises:
            TypeError: If x is not an Expression.
        """
        super().__init__(
            exprs={"x": x},
            buffers={"l1_scaling": l1_scaling, "l2_scaling": l2_scaling},
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
        """Returns True if the input is a Variable."""
        input_ = self.get_input("x")
        if isinstance(input_, Variable):
            return True
        return False

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

    def decompose(self) -> list[AtomDecomposition] | None:
        """Decompose the ElasticNet atom if its input is an affine expression.

        Returns:
            list[AtomDecomposition] | None: Decomposition containing the new atom r(z)
                and affine expression if decomposable, None otherwise.
        """
        # If the input expression is not affine, cannot decompose
        input_expr = self.get_input("x")
        if not input_expr.is_affine():
            return None

        new_var = Variable.like(input_expr)

        l1_scaling = self.get_buffer("l1_scaling")
        l2_scaling = self.get_buffer("l2_scaling")
        new_atom = ElasticNet(new_var, l1_scaling=l1_scaling, l2_scaling=l2_scaling)

        return [AtomDecomposition(atom=new_atom, affine_expr=input_expr)]

    def _scale(self, scaling: float) -> Self:
        """Scale the elastic net regularization atom."""
        new_l1 = self.get_buffer("l1_scaling") * scaling
        new_l2 = self.get_buffer("l2_scaling") * scaling
        return ElasticNet(self.get_input("x"), l1_scaling=new_l1, l2_scaling=new_l2)
