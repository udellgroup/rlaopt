"""Base class for optimization atoms."""

from abc import ABC, abstractmethod

import torch
from typing_extensions import Self

from rlaopt.expression import Expression, ExprTree, Variable
from rlaopt.ext_tensordict import TensorDict


class Atom(Expression, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable) and
    can be composed to form more complex objective functions.

    Atoms extend Expression with:
        - Input registration system for Variables and Expressions
        - Buffer management for constants and hyperparameters

    Subclasses must implement:
        - is_smooth() - whether the function is differentiable everywhere
        - is_proxable() - whether the proximal operator is computable
        - forward() - evaluation of the atom
        - _prox() - prox operator of the atom
    """

    def __init__(
        self,
        exprs: dict[str, Expression],
        buffers: dict[str, torch.Tensor | float | None],
        variable_names: list[str] | None = None,
    ):
        """Initialize the atom.

        Subclasses should call this constructor to ensure proper initialization.
        """
        super().__init__()

        # Automatically register all input expressions
        variable_names = variable_names or []
        variable_only = {var_name: True for var_name in variable_names}
        for name, expr in exprs.items():
            self._register_input(name, expr, variable_only.get(name, False))

        # Automatically register all buffers
        for name, buffer in buffers.items():
            self._register_atom_buffer(name, buffer)

    @abstractmethod
    def is_proxable(self) -> bool:
        """Check if the expression has a computable proximal operator.

        The proximal operator is used in proximal gradient methods and ADMM
        for non-smooth optimization. An expression is proxable if its proximal
        operator can be computed efficiently in closed form.

        Returns:
            bool: True if the expression is proxable, False otherwise.
        """
        pass

    def prox(self, variable_values: TensorDict, prox_scaling: float) -> TensorDict:
        """Proximal operator corresponding to the atom.

        Args:
            variable_values: TensorDict containing the location at which to evaluate
                the proximal operator.
            prox_scaling: Scaling parameter for the proximal operator.

        Returns:
            TensorDict: Result of applying the proximal operator at the given
                variable_values. Note that the returned TensorDict may contain only a
                subset of the variables in variable_values, depending on which variables
                are relevant to the atom.

        Raises:
            NotImplementedError: If the atom is not proxable.
        """
        if not self.is_proxable():
            raise NotImplementedError(
                f"Proximal operator for {self.__class__.__name__} with given inputs "
                "is not implemented."
            )

        relevant_vars = self.select_relevant_variables(variable_values)
        prox_result = self._prox(relevant_vars, prox_scaling)
        return prox_result

    @abstractmethod
    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Proximal operator implementation for the atom."""
        pass

    def get_input(self, name: str) -> Expression:
        """Retrieve a registered input expression by name.

        Args:
            name: Name of the input expression to retrieve.

        Returns:
            Expression: The registered input expression.

        Raises:
            KeyError: If no input with the given name exists.
        """
        if not hasattr(self, name):
            raise KeyError(f"No input expression named '{name}' found.")
        return getattr(self, name)

    def _register_atom_buffer(self, name: str, buffer):
        """Register a buffer (non-trainable constant) with the atom.

        Buffers store constants, hyperparameters, or fixed data that should be
        tracked by the module but not optimized. Unlike parameters, buffers do
        not receive gradients.

        Args:
            name: Name for the buffer.
            buffer: Value to register (float, Parameter, or Tensor).

        """
        if isinstance(buffer, float):
            self.register_buffer(name, torch.tensor(float(buffer)))
        elif isinstance(buffer, torch.Tensor):
            self.register_buffer(name, buffer)
        elif buffer is None:
            self.register_buffer(name, None)
        else:
            raise TypeError(
                f"Expected float, Tensor, or None, but got {type(buffer).__name__}"
            )

    def _register_input(self, name: str, x: Expression, variable_only: bool):
        """Register an input (Expression) with the atom."""
        if variable_only and not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, but got {type(x).__name__} instead.")

        if not isinstance(x, Expression):
            raise TypeError(f"Expected Expression, but got {type(x).__name__}")

        self.add_module(name, x)

    def _scale(self, scaling: float) -> Self:
        """Scale the atom by a scalar constant.

        This method should be overridden by subclasses that support scalar
        multiplication. The default implementation returns NotImplemented.

        Args:
            scaling: Scalar value to multiply the atom by.

        Returns:
            Self: A new atom scaled by the given value, or NotImplemented if
                scalar multiplication is not supported.
        """
        return NotImplemented

    def tree(self) -> ExprTree:
        """Return tree representation for Atom.

        If the atom has input expressions, includes them in the tree.
        Otherwise, returns just the atom class name as a leaf node.

        Returns:
            ExprTree: Tree with atom class name and optional input child.
        """
        input_expr_trees = [
            expr.tree()
            for _, expr in self.named_children()
            if isinstance(expr, Expression)
        ]
        if len(input_expr_trees) == 0:
            return ExprTree(self.__class__.__name__)
        return ExprTree(self.__class__.__name__, *input_expr_trees)
