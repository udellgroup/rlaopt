"""Base class for optimization atoms."""

from abc import ABC, abstractmethod

import torch
from typing_extensions import Self

from rlaopt.expression import Expression, Variable
from rlaopt.expression.tree import ExprTree
from rlaopt.utils.counter import Counter


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

    Examples:
        >>> class L1Norm(AtomExpression):
        ...     def __init__(self, x: Variable, scaling: float = 1.0):
        ...         super().__init__()
        ...         self.register_input(x)
        ...         self.register_atom_buffer("scaling", scaling)
        ...
        ...     def forward(self) -> torch.Tensor:
        ...         value = self[1]
        ...         return self.scaling * torch.sum(torch.abs(value))
    """

    def __init__(self):
        """Initialize the atom.

        Subclasses should call this constructor to ensure proper initialization.
        Subclasses should also register any variables, expressions, or buffers
        they use with the appropriate registration methods.
        """
        super().__init__()
        self._exprs_names = {}
        self._ids_to_exprs = {}
        self._expr_counter = Counter()

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

    def get_input(self, input_registration_id: int) -> Expression:
        """Fetches registered input expression from its ID.

        Given an input registration ID, retrieves the corresponding
        registered Expression associated with the atom.
        The input registration ID corresponds to the order in which
        inputs were registered using the `register_input` method.
        I.e., the first registered input has ID 1, the second has ID 2, etc.
        """
        if input_registration_id not in self.ids_to_exprs:
            raise KeyError(
                f"No input registered with ID {input_registration_id}. "
                f"Valid IDs are: {sorted(self.ids_to_exprs.keys())}"
            )
        return self._ids_to_exprs[input_registration_id]

    def register_atom_buffer(self, name: str, buffer):
        """Register a buffer (non-trainable constant) with the atom.

        Buffers store constants, hyperparameters, or fixed data that should be
        tracked by the module but not optimized. Unlike parameters, buffers do
        not receive gradients.

        Args:
            name: Name for the buffer.
            buffer: Value to register (float, Parameter, or Tensor).

        Examples:
            >>> class ScaledNorm(AtomExpression):
            ...     def __init__(self, x: Variable, scaling: float):
            ...         super().__init__()
            ...         self.register_variable(x)
            ...         self.register_atom_buffer("scaling", scaling)
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

    def register_input(self, x: Expression, variable_only: bool = False):
        """Register an input (Expression) with the atom.

        This is a convenience method that automatically determines whether the
        input is a Expression and registers appropriately.

        Args:
            x: Input to register (Expression).
            variable_only: If True, only allow Variable inputs (default: False).

        Raises:
            TypeError: If x is not a Variable when variable_only is True.
            TypeError: If x is not an Expression.
        """
        if variable_only and not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, but got {type(x).__name__} instead.")

        if not isinstance(x, Expression):
            raise TypeError(f"Expected Expression, but got {type(x).__name__}")

        # We don't allow registration of variable
        # with the same name as an existing variable.
        if isinstance(x, Variable):
            expr_name = x.name
            if expr_name in self._exprs_names:
                raise ValueError(f"Variable '{expr_name}' is already registered.")
            unique_name = expr_name
            # Variables get count of 1
            self._exprs_names[expr_name] = 1
        else:
            # Use class name as base_expr_name
            base_expr_name = x._get_name()
            # Get count of how many times class of type
            # base_expr_name has been registered
            count = self._exprs_names.get(base_expr_name, 0)

            if count == 0:
                # First occurrence - use base name
                unique_name = base_expr_name
            else:
                # Collision - append count to get
                # a unique name for the expression
                unique_name = f"{base_expr_name}_{count}"

            # Update count
            self._exprs_names[base_expr_name] = count + 1

        # Get input ID and update counter
        expr_id = self.expr_count
        self._expr_counter.count += 1
        # Register input expression
        self._ids_to_exprs[expr_id] = x
        self.add_module(unique_name, x)

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
        if self.ids_to_exprs:
            # Get expression trees for all the input expressions used to
            # construct the atom.
            input_expr_trees = [
                self.ids_to_exprs[expr_id].tree() for expr_id in self.ids_to_exprs
            ]
            return ExprTree(self.__class__.__name__, *input_expr_trees)
        return ExprTree(self.__class__.__name__)

    @property
    def expr_count(self) -> int:
        """Returns the number of registered expressions with the atom."""
        return self._expr_counter.count

    @property
    def expr_names(self) -> dict[str, int]:
        """Returns a dictionary mapping expression names to their occurrence counts.

        For Variables, maps the variable name to 1. For other Expressions, maps
        the base class name (e.g., 'SumSquares') to the count of how many times
        that type has been registered. This count is used to generate unique names
        for duplicate expression types (e.g., 'SumSquares_1', 'SumSquares_2').

        Returns:
            dict[str, int]: Dictionary mapping expression names to counts.
        """
        return self._exprs_names

    @property
    def ids_to_exprs(self) -> dict[int, Expression]:  # Changed return type annotation
        """Returns a mapping of registered expression IDs to their Expression objects."""
        return self._ids_to_exprs

    def __getitem__(self, idx: int) -> Expression:
        """Magic method for fetching expression from its idx ID."""
        return self.get_input(idx)
