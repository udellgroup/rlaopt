"""Base class for optimization atoms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch

from rlaopt.expression import Expression, Variable


class InputType(Enum):
    """Type of input registered with an atom.

    Attributes:
        VARIABLE: Input is a Variable (leaf parameter).
        EXPRESSION: Input is a composite Expression.
    """

    VARIABLE = "variable"
    EXPRESSION = "expression"


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
        - is_subsamplable() - whether the atom supports data subsampling
        - subsample() - create a subsampled version of the atom
        - to_cvxpy() - conversion to CVXPY representation

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
        descent or stochastic ADMM.

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

    def get_variable(self, var_name: str) -> torch.nn.Parameter:
        """Retrieve a registered variable by name.

        Args:
            var_name: Name of the variable to retrieve.

        Returns:
            torch.nn.Parameter: The parameter tensor for the variable.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> atom = SomeAtom(x)
            >>> param = atom.get_variable('x')
        """
        return getattr(self, var_name)

    def register_atom_buffer(
        self, name: str, buffer: float | torch.nn.Parameter | torch.Tensor
    ):
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
        if isinstance(buffer, torch.Tensor):
            self.register_buffer(name, buffer)
        elif isinstance(buffer, torch.nn.Parameter):
            self.register_buffer(name, buffer.data)
        else:
            self.register_buffer(name, torch.tensor(float(buffer)))

    def register_input(self, x: Variable | Expression):
        """Register an input (Variable or Expression) with the atom.

        This is a convenience method that automatically determines whether the
        input is a Variable or Expression and calls the appropriate registration
        method. Sets self.input_type to indicate which type was registered.

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
        self.input_type = self.expr_type(x)
        if self.input_type == InputType.VARIABLE:
            self.register_variable(x)
        elif self.input_type == InputType.EXPRESSION:
            self.register_expression(x)
        else:
            raise TypeError(f"Expected Variable or Expression, got {type(x)}")

    def register_variable(self, x: Variable):
        """Register a Variable with the atom.

        Registers the variable's parameter so it can be optimized and tracked
        by the module. Stores the variable's name in self.var_name for later
        retrieval.

        Args:
            x: Variable to register.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> atom.register_variable(x)
            >>> atom.var_name
            'x'
        """
        self._var_name = x.name
        self.register_parameter(self._var_name, x.value)

    def register_expression(self, expr: Expression):
        """Register an Expression as a submodule of the atom.

        Registers the expression as a submodule so its parameters are tracked
        and gradients flow correctly. Stores the module name in self.module_name
        for later retrieval.

        Args:
            expr: Expression to register.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> y = Variable((5,), name='y')
            >>> expr = x + y
            >>> atom.register_expression(expr)
            >>> atom.module_name
            'AddExpression'
        """
        self._expr_name = expr._get_name()
        self.add_module(self._expr_name, expr)

    @staticmethod
    def expr_type(x: Variable | Expression) -> InputType:
        """Determine the type of the input.

        Args:
            x: Input to check.

        Returns:
            InputType: InputType.VARIABLE if x is a Variable,
                      InputType.EXPRESSION if x is an Expression.

        Raises:
            TypeError: If x is neither a Variable nor an Expression.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> AtomExpression.expr_type(x)
            <InputType.VARIABLE: 'variable'>

            >>> expr = x + y
            >>> AtomExpression.expr_type(expr)
            <InputType.EXPRESSION: 'expression'>
        """
        if isinstance(x, Variable):
            return InputType.VARIABLE
        else:
            return InputType.EXPRESSION

    @property
    def var_name(self):
        """Get the registered variable's name."""
        return getattr(self, "_var_name", None)

    @property
    def expr_name(self) -> Expression:
        """Get the expression registered with the atom."""
        return getattr(self, "_expr_name", None)
