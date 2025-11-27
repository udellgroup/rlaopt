"""Module for Constant class."""

import torch

from rlaopt.expression.expression import Expression
from rlaopt.expression.tree import ExprTree


class Constant(Expression):
    """Constant value expression.

    Represents a constant (non-trainable) value in an expression tree.
    Constants are stored as buffers rather than parameters, so they
    don't receive gradients and don't appear in parameter optimization.

    Args:
        value: The constant value (float, int, or torch.Tensor).

    Attributes:
        _value: The constant value stored as a buffer.

    Examples:
        >>> c = Constant(3.14)
        >>> c.forward()
        tensor(3.1400)
        >>> c2 = Constant(torch.ones(5))
        >>> c2.forward().shape
        torch.Size([5])
    """

    def __init__(self, value: float | int | torch.Tensor):
        """Initialize a constant expression.

        Args:
            value: The constant value to store.
        """
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        # register as buffer so it is visible but not a parameter
        self.register_buffer("_value", value)

    @property
    def value(self):
        """Get the constant value.

        Returns:
            torch.Tensor: The constant value.
        """
        return getattr(self, "_value")

    def is_smooth(self) -> bool:
        """Constants are smooth (trivially differentiable).

        Returns:
            bool: Always True.
        """
        return True

    def forward(self) -> torch.Tensor:
        """Evaluate the constant (returns itself).

        Returns:
            torch.Tensor: The constant value.
        """
        return self.value

    def tree(self) -> ExprTree:
        """Return tree representation for Constant (leaf node).

        Returns:
            ExprTree: Leaf node with type 'Constant'.
        """
        return ExprTree("Constant")

    def is_affine(self) -> bool:
        """Constants are affine expressions.

        Returns:
            bool: Always True.
        """
        return True

    def __neg__(self):
        """Negate the constant (keeps it as a constant).

        Returns:
            Constant: Negated constant.
        """
        return Constant(-self.value)
