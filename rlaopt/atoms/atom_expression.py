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
        ...         super().__init__(x, variables_only=True, scaling=scaling)
        ...
        ...     def forward(self) -> torch.Tensor:
        ...         value = self.x1.forward()
        ...         return self.scaling * torch.sum(torch.abs(value))
    """

    def __init__(
        self,
        *exprs: tuple[Expression],
        variable_only: bool = False,
        **buffers: dict[str, torch.Tensor],
    ):
        """Initialize the atom.

        Subclasses should call this constructor to ensure proper initialization.
        Subclasses should also register any variables, expressions, or buffers
        they use with the appropriate registration methods.
        """
        super().__init__()
        self._variable_only = variable_only
        self._var_names = set()  # Track registered variable names
        self._expr_counter = Counter()

        # Automatically register all input expressions
        for expr in exprs:
            self._register_input(expr)

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

    def _register_input(self, x: Expression):
        """Register an input (Expression) with the atom."""
        if self.variable_only and not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, but got {type(x).__name__} instead.")

        if not isinstance(x, Expression):
            raise TypeError(f"Expected Expression, but got {type(x).__name__}")

        # Check for duplicate variable names
        if isinstance(x, Variable):
            if x.name in self._var_names:
                raise ValueError(f"Variable '{x.name}' is already registered.")
            self._var_names.add(x.name)

        # Get input ID and update counter
        expr_id = self.expr_count
        self._expr_counter.count += 1

        # Register with simple x_i naming
        expr_name = f"x{expr_id}"
        self.add_module(expr_name, x)

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
        if self.expr_count > 0:
            # Get expression trees for all the input expressions used to
            # construct the atom.
            input_expr_trees = [
                getattr(self, f"x{i}").tree() for i in range(1, self.expr_count)
            ]
            return ExprTree(self.__class__.__name__, *input_expr_trees)
        return ExprTree(self.__class__.__name__)

    @property
    def expr_count(self) -> int:
        """Returns the number of registered expressions with the atom."""
        return self._expr_counter.count

    @property
    def variable_only(self) -> bool:
        """Returns whether atom supports only variable registration."""
        return self._variable_only
