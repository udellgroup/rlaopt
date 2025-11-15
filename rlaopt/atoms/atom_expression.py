"""Base class for optimization atoms."""

from abc import ABC, abstractmethod

import torch
from typing_extensions import Self

from rlaopt.expression import Expression, Variable
from rlaopt.expression.tree import ExprTree


class AtomExpression(Expression, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, subsamplable) and
    can be composed to form more complex objective functions.

    Atoms extend Expression with:
        - Input registration system for Variables and Expressions
        - Buffer management for constants and hyperparameters
        - Subsampling support for stochastic optimization methods

    Subclasses must implement:
        - is_smooth() - whether the function is differentiable everywhere
        - is_proxable() - whether the proximal operator is computable
        - forward() - evaluation of the atom
        - prox() - prox operator of the atom
        - is_subsamplable() - whether the atom supports data subsampling
        - subsample() - create a subsampled version of the atom
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
    def is_subsamplable(self) -> bool:
        """Check if the atom supports subsampling.

        Subsampling allows the atom to operate on a subset of data, which is
        essential for stochastic optimization methods like mini-batch gradient
        descent.

        Returns:
            bool: True if the atom supports subsampling, False otherwise.

        Examples:
            >>> # Loss functions with data typically support subsampling
            >>> loss = LinearRegression(dataloader, beta)
            >>> loss.is_subsamplable()
            True

            >>> # Regularizers typically don't support subsampling
            >>> reg = L1Norm(beta)
            >>> reg.is_subsamplable()
            False
        """
        pass

    @abstractmethod
    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Proximal operator corresponding to the atom."""
        pass

    @abstractmethod
    def subsample(self, indices: torch.Tensor) -> Self:
        """Create a subsampled version of the atom.

        This method should only be called if the atom is subsamplable.
        The returned atom operates only on the data indexed by the provided indices.

        Args:
            indices: Indices of data points to include in the subsample.

        Returns:
            Self: New atom representing the subsampled version.

        Raises:
            NotImplementedError: If the atom does not support subsampling.

        Examples:
            >>> loss = LinearRegression(dataloader, beta)
            >>> subset_indices = torch.tensor([0, 5, 10, 15])
            >>> mini_batch_loss = loss.subsample(subset_indices)
        """
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
        """Return tree representation for AtomExpression.

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
