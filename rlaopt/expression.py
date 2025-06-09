from abc import ABC, abstractmethod

import cvxpy as cp
import torch


class Expression(torch.nn.Module, ABC):
    """Base class for mathematical expressions in optimization problems."""

    def __init__(self):
        """Initialize the expression."""
        super().__init__()

    def forward(self, location=None):
        """Evaluate the expression, potentially at a specific location.

        Args:
            location: Optional location at which to evaluate

        Returns:
            The evaluated expression
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def to_cvxpy(self) -> cp.Expression:
        """Convert the expression to a CVXPY expression.

        Returns:
            A CVXPY expression representing this mathematical expression.
        """
        pass

    def __call__(self, location=None):
        """Call the expression, optionally at a specific location."""
        return self.forward(location)

    def __add__(self, other):
        return AddExpression(self, other)

    def __radd__(self, other):
        return AddExpression(other, self)

    def __mul__(self, other):
        return MulExpression(self, other)

    def __rmul__(self, other):
        return MulExpression(other, self)

    def __neg__(self):
        return MulExpression(-1.0, self)

    def __sub__(self, other):
        return AddExpression(self, -other)

    def __rsub__(self, other):
        return AddExpression(other, -self)


class AddExpression(Expression):
    """Expression representing addition."""

    def __init__(self, left, right):
        super().__init__()

        # Register based on type, using clear distinctive names
        if isinstance(left, Expression):
            self.add_module("left", left)
        elif isinstance(left, (int, float)):
            self.register_buffer("left", torch.tensor(float(left)))
        elif isinstance(left, torch.Tensor):
            self.register_buffer("left", left)
        else:
            raise TypeError(f"Unsupported type for left term: {type(left)}")

        if isinstance(right, Expression):
            self.add_module("right", right)
        elif isinstance(right, (int, float)):
            self.register_buffer("right", torch.tensor(float(right)))
        elif isinstance(right, torch.Tensor):
            self.register_buffer("right", right)
        else:
            raise TypeError(f"Unsupported type for right term: {type(right)}")

    def forward(self, location=None):
        """Evaluate the addition, potentially at a specific location.

        Args:
            location: Optional location at which to evaluate

        Returns:
            Sum of the left and right terms
        """
        # Get left value from the appropriate registered attribute
        if hasattr(self, "left") and callable(self.left):
            left_value = self.left(location) if location is not None else self.left()
        else:
            left_value = self.left

        # Get right value from the appropriate registered attribute
        if hasattr(self, "right") and callable(self.right):
            right_value = self.right(location) if location is not None else self.right()
        else:
            right_value = self.right

        return left_value + right_value

    def to_cvxpy(self):
        left_cvxpy = (
            self.left.to_cvxpy()
            if isinstance(self.left, Expression)
            else self.left.numpy(force=True)
        )
        right_cvxpy = (
            self.right.to_cvxpy()
            if isinstance(self.right, Expression)
            else self.right.numpy(force=True)
        )

        return left_cvxpy + right_cvxpy


class MulExpression(Expression):
    """Expression representing multiplication."""

    def __init__(self, left, right):
        super().__init__()

        # Do not allow both left and right to be Expression instances
        if isinstance(left, Expression) and isinstance(right, Expression):
            raise TypeError("Cannot multiply two Expression instances directly.")

        # Register based on type, using clear distinctive names
        if isinstance(left, Expression):
            self.add_module("left", left)
        elif isinstance(left, (int, float)):
            self.register_buffer("left", torch.tensor(float(left)))
        elif isinstance(left, torch.Tensor):
            self.register_buffer("left", left)
        else:
            raise TypeError(f"Unsupported type for left term: {type(left)}")

        if isinstance(right, Expression):
            self.add_module("right", right)
        elif isinstance(right, (int, float)):
            self.register_buffer("right", torch.tensor(float(right)))
        elif isinstance(right, torch.Tensor):
            self.register_buffer("right", right)
        else:
            raise TypeError(f"Unsupported type for right term: {type(right)}")

    def forward(self, location=None):
        """Evaluate the multiplication, potentially at a specific location.

        Args:
            location: Optional location at which to evaluate

        Returns:
            Product of the left and right terms
        """
        # Get left value from the appropriate registered attribute
        if hasattr(self, "left") and callable(self.left):
            left_value = self.left(location) if location is not None else self.left()
        else:
            left_value = self.left

        # Get right value from the appropriate registered attribute
        if hasattr(self, "right") and callable(self.right):
            right_value = self.right(location) if location is not None else self.right()
        else:
            right_value = self.right

        return left_value * right_value

    def to_cvxpy(self):
        left_cvxpy = (
            self.left.to_cvxpy()
            if isinstance(self.left, Expression)
            else self.left.numpy(force=True)
        )
        right_cvxpy = (
            self.right.to_cvxpy()
            if isinstance(self.right, Expression)
            else self.right.numpy(force=True)
        )

        return left_cvxpy * right_cvxpy
