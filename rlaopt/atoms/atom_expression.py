"""Base class for optimization atoms."""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch

from rlaopt.expression.expression import Expression, Variable


class AtomExpression(Expression, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, etc.) and can be composed
    to form more complex objective functions.
    """

    def __init__(self):
        """Initializes the atom.

        Subclasses should call this constructor to ensure proper initialization.
        Subclasses should also register any variables they use with the atom.
        """
        super().__init__()

    @abstractmethod
    def is_subsamplable(self) -> bool:
        """Returns True if the atom supports subsampling (e.g., for stochastic
        methods)."""
        pass

    @abstractmethod
    def subsample(self, indices: torch.Tensor) -> AtomExpression:
        """Returns a subsampled version of the atom.

        This method should only be called if the atom is subsamplable.
        Otherwise, it should raise a NotImplementedError.

        Args:
            indices: Indices to subsample

        Returns:
            New atom representing the subsampled version

        Raises:
            NotImplementedError: If the atom does not support subsampling.
        """
        pass

    def get_variable(self, var_name: str) -> torch.nn.Parameter:
        return getattr(self, var_name)

    def register_atom_buffer(
        self, name: str, buffer: float | torch.nn.Parameter | torch.Tensor
    ):
        """Registers a buffer with the atom.

        This method should be used by subclasses to register any buffers they
        use. It ensures that the buffer is also registered with the parent
        Expression class.

        Args:
            name: Name of the buffer
            buffer: Buffer to register (float, torch.nn.Parameter, or torch.Tensor)
        """
        if isinstance(buffer, torch.Tensor):
            self.register_buffer(name, buffer)
        elif isinstance(buffer, torch.nn.Parameter):
            self.register_buffer(name, buffer.data)
        else:
            self.register_buffer(name, torch.tensor(float(buffer)))

    def register_input(self, x: Variable | Expression):
        """Registers an input (variable or expression) with the atom.

        This method should be used by subclasses to register any inputs they
        use. It ensures that the input is also registered with the parent
        Expression class.

        Args:
            x: Input to register (Variable or Expression)
        """
        self.input_type = self.expr_type(x)
        if self.input_type == "variable":
            self.register_variable(x)
        elif self.input_type == "expression":
            self.register_expression(x)
        else:
            raise TypeError(f"Expected Variable or Expression, got {type(x)}")

    def register_variable(self, x: Variable):
        """Registers a variable with the atom.

        This method should be used by subclasses to register any variables they
        use. It ensures that the variable is also registered with the parent
        Expression class.

        Args:
            x: Variable to register
        """
        self.var_name = x.name
        self.register_parameter(self.var_name, x.value)

    def register_expression(self, expr: Expression):
        """Registers an expression with the atom.

        This method should be used by subclasses to register any expressions they
        use. It ensures that the expression is also registered with the parent
        Expression class.

        Args:
            expr: Expression to register
        """
        self.module_name = expr._get_name()
        self.add_module(self.module_name, expr)

    @staticmethod
    def expr_type(x: Variable | Expression) -> str:
        """Determines the type of the input (variable or expression).

        Args:
            x: Input to check

        Returns:
            "variable" if x is a Variable, "expression" if x is an Expression

        Raises:
            TypeError: If x is neither a Variable nor an Expression
        """
        if isinstance(x, Variable):
            return "variable"
        elif isinstance(x, Expression):
            return "expression"
        else:
            raise TypeError(f"Expected Variable or Expression, got {type(x)}")
