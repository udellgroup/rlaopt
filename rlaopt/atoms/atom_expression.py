"""Base class for optimization atoms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from rlaopt.expression import Expression, Variable


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
        ...         self.register_variable(x)
        ...         self.register_atom_buffer("scaling", scaling)
        ...
        ...     def forward(self) -> torch.Tensor:
        ...         value = self.get_variable(self.var_name)
        ...         return self.scaling * torch.sum(torch.abs(value))
    """

    def __init__(self):
        """Initialize the atom.

        Subclasses should call this constructor to ensure proper initialization.
        Subclasses should also register any variables, expressions, or buffers
        they use with the appropriate registration methods.
        """
        super().__init__()

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
    def subsample(self, indices: torch.Tensor) -> AtomExpression:
        """Create a subsampled version of the atom.

        This method should only be called if the atom is subsamplable.
        The returned atom operates only on the data indexed by the provided indices.

        Args:
            indices: Indices of data points to include in the subsample.

        Returns:
            AtomExpression: New atom representing the subsampled version.

        Raises:
            NotImplementedError: If the atom does not support subsampling.

        Examples:
            >>> loss = LinearRegression(dataloader, beta)
            >>> subset_indices = torch.tensor([0, 5, 10, 15])
            >>> mini_batch_loss = loss.subsample(subset_indices)
        """
        pass

    def get_input(self) -> Expression | torch.nn.Parameter:
        """Returns input variable or expression used to construct the Atom."""
        if self.var_name:
            return getattr(self, self.var_name)
        else:
            return getattr(self, self.expr_name)

    def register_atom_buffer(self, name: str, buffer) -> torch.nn.Buffer:
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
        elif isinstance(buffer, torch.nn.Parameter):
            self.register_buffer(name, buffer.data)
        elif buffer is None:
            self.register_buffer(name, None)
        else:
            raise TypeError(
                "Expected float, Tensor, Parameter, or None, but got "
                f"{type(buffer).__name__}"
            )

    def register_input(self, x):
        """Register an input (Variable or Expression) with the atom.

        This is a convenience method that automatically determines whether the
        input is a Variable or Expression and registers then register appropriately.

        Args:
            x: Input to register (Variable or Expression).

        Raises:
            TypeError: If x is neither a Variable nor an Expression.

        Examples:
            >>> # Registers a Variable
            >>> x = Variable((5,), name='x')
            >>> atom.register_input(x)  # Sets input_type to InputType.VARIABLE

            >>> # Registers an Expression
            >>> expr = x + y
            >>> atom.register_input(expr)  # Sets input_type to InputType.EXPRESSION
        """
        if isinstance(x, Variable):
            self._var_name = x.name
            self.register_parameter(self._var_name, x.value)
        elif isinstance(x, Expression):
            self._expr_name = x._get_name()
            self.add_module(self._expr_name, x)
        else:
            raise TypeError(
                f"Expected Expression or Variable, but got {type(x).__name__}"
            )

    @property
    def var_name(self) -> str | None:
        """Get the registered variable's name."""
        return getattr(self, "_var_name", None)

    @property
    def expr_name(self) -> Expression | None:
        """Get the expression registered with the atom."""
        return getattr(self, "_expr_name", None)
