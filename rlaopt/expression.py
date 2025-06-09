from abc import ABC, abstractmethod
import cvxpy as cp
import torch


class Expression(torch.nn.Module, ABC):
    """Base class for mathematical expressions in optimization problems."""

    def __init__(self):
        """Initialize the expression."""
        super().__init__()

    def forward(self):
        """Evaluate the expression using registered variables.

        Returns:
            The evaluated expression
        """
        # Forward simply calls evaluate_at with no substitutions
        return self.evaluate_at()

    @abstractmethod
    def evaluate_at(self, **variable_locations):
        """Evaluate the expression at specific variable locations.

        Args:
            **variable_locations: Mapping of variable names to locations

        Returns:
            The evaluated expression with variables substituted with their locations
        """
        pass

    @abstractmethod
    def to_cvxpy(self) -> cp.Expression:
        """Convert the expression to a CVXPY expression.

        Returns:
            A CVXPY expression representing this mathematical expression.
        """
        pass

    def __call__(self, **variable_locations):
        """Call the expression, choosing the appropriate evaluation method.

        If called with no arguments, uses forward(). If called with keyword arguments,
        uses evaluate_at().
        """
        if not variable_locations:
            return self.forward()
        else:
            return self.evaluate_at(**variable_locations)

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

    def evaluate_at(self, **variable_locations):
        """Evaluate the addition at specific variable locations."""
        # Get left value with substitutions
        if hasattr(self, "left") and isinstance(self.left, Expression):
            left_value = self.left.evaluate_at(**variable_locations)
        else:
            left_value = self.left

        # Get right value with substitutions
        if hasattr(self, "right") and isinstance(self.right, Expression):
            right_value = self.right.evaluate_at(**variable_locations)
        else:
            right_value = self.right

        return left_value + right_value

    def to_cvxpy(self):
        """Convert to a CVXPY expression."""
        left_cvxpy = (
            self.left.to_cvxpy()
            if isinstance(self.left, Expression)
            else float(self.left.item())
            if self.left.numel() == 1
            else self.left.numpy()
        )
        right_cvxpy = (
            self.right.to_cvxpy()
            if isinstance(self.right, Expression)
            else float(self.right.item())
            if self.right.numel() == 1
            else self.right.numpy()
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

    def evaluate_at(self, **variable_locations):
        """Evaluate the multiplication at specific variable locations."""
        # Get left value with substitutions
        if hasattr(self, "left") and isinstance(self.left, Expression):
            left_value = self.left.evaluate_at(**variable_locations)
        else:
            left_value = self.left

        # Get right value with substitutions
        if hasattr(self, "right") and isinstance(self.right, Expression):
            right_value = self.right.evaluate_at(**variable_locations)
        else:
            right_value = self.right

        return left_value * right_value

    def to_cvxpy(self):
        """Convert to a CVXPY expression."""
        left_cvxpy = (
            self.left.to_cvxpy()
            if isinstance(self.left, Expression)
            else float(self.left.item())
            if self.left.numel() == 1
            else self.left.numpy()
        )
        right_cvxpy = (
            self.right.to_cvxpy()
            if isinstance(self.right, Expression)
            else float(self.right.item())
            if self.right.numel() == 1
            else self.right.numpy()
        )

        return left_cvxpy * right_cvxpy
