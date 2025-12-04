"""Base class for nonsmooth regularization atoms."""

from abc import ABC

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom, AtomDecomposition
from rlaopt.expression import Expression, Variable


class NonsmoothRegularizer(Atom, ABC):
    """Base class for nonsmooth regularization atoms.

    Provides common functionality for regularizers like L1Norm, NucNorm, and
    ElasticNet that:
    - Are nonsmooth (is_smooth() = False)
    - Are proxable only when the input is a Variable
    - Can be scaled by multiplying their scaling parameters
    - Can be decomposed when the input is an affine expression
    """

    def __init__(
        self,
        x: Expression,
        scaling_params: dict[str, float | torch.Tensor],
    ):
        """Initialize the nonsmooth regularizer.

        Args:
            x: Expression to regularize.
            scaling_params: Dictionary of scaling parameter names to values.
        """
        super().__init__(exprs={"x": x}, buffers=scaling_params)
        # Track which buffers are scaling parameters for _scale and decompose
        self._scaling_param_names = list(scaling_params.keys())

    def is_smooth(self) -> bool:
        """Check if the regularizer is smooth.

        Returns:
            bool: Always False, as these are nonsmooth regularizers.
        """
        return False

    def is_proxable(self) -> bool:
        """Check if the regularizer has a computable proximal operator.

        Returns:
            bool: True if the input is a Variable, False otherwise.
        """
        input_ = self.get_input("x")
        return isinstance(input_, Variable)

    def decompose(self) -> list[AtomDecomposition] | None:
        """Decompose the regularizer if its input is an affine expression.

        Creates a new variable z matching the input expression's properties,
        constructs a new regularizer with the same scaling parameters applied
        to z, and returns the decomposition.

        Returns:
            list[AtomDecomposition] | None: Decomposition containing the new
                regularizer and affine expression if decomposable, None otherwise.
        """
        input_expr = self.get_input("x")

        # Check if input is affine
        if not input_expr.is_affine():
            return None

        # Create new variable matching the input expression
        new_var = Variable.like(input_expr)

        # Get all scaling parameters
        scaling_params = {
            name: self.get_buffer(name) for name in self._scaling_param_names
        }

        # Reconstruct the same type of regularizer with new variable
        atom_class = type(self)
        new_atom = atom_class(new_var, **scaling_params)

        return [AtomDecomposition(atom=new_atom, affine_expr=input_expr)]

    def _scale(self, scaling: float) -> Self:
        """Scale the regularizer by multiplying all scaling parameters.

        Args:
            scaling: Scalar value to multiply all scaling parameters by.

        Returns:
            Self: A new regularizer with scaled parameters.
        """
        # Scale all scaling parameters
        scaled_params = {
            name: self.get_buffer(name) * scaling for name in self._scaling_param_names
        }

        # Reconstruct with same input but scaled parameters
        atom_class = type(self)
        return atom_class(self.get_input("x"), **scaled_params)
