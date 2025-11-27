"""Unary expression operations (reduce sum, transpose, power, etc.)."""

from abc import ABC, abstractmethod

import torch

from rlaopt.expression.expression import Expression
from rlaopt.expression.tree import ExprTree
from rlaopt.expression.utils import _to_expr


class _Unary(Expression, ABC):
    """Base class for unary operations on expressions.

    All unary operations have a single operand and apply some transformation.
    Subclasses must implement:
    - _forward_op(): The actual operation to apply
    - _is_affine_preserving(): Whether the operation preserves affineness
    - _operation_name(): Name of the operation for tree representation

    Args:
        operand: Expression to apply operation to.
    """

    def __init__(self, operand: Expression | float | int | torch.Tensor):
        """Initialize unary expression.

        Args:
            operand: Expression or value to operate on.
        """
        super().__init__()
        self.operand = _to_expr(operand)
        self.add_module("_operand", self.operand)

    def forward(self) -> torch.Tensor:
        """Evaluate the unary operation.

        Returns:
            torch.Tensor: Result of applying operation to operand.
        """
        val = self.operand.forward()
        return self._forward_op(val)

    @abstractmethod
    def _forward_op(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the specific operation to the tensor.

        Args:
            tensor: Evaluated operand tensor.

        Returns:
            torch.Tensor: Result of the operation.
        """
        pass

    def is_smooth(self) -> bool:
        """Most unary operations preserve smoothness.

        Returns:
            bool: True if operand is smooth.
        """
        return self.operand.is_smooth()

    def is_affine(self) -> bool:
        """Affine if operation preserves affineness and operand is affine.

        Returns:
            bool: True if operation preserves affineness and operand is affine.
        """
        return self._is_affine_preserving() and self.operand.is_affine()

    @abstractmethod
    def _is_affine_preserving(self) -> bool:
        """Whether this operation preserves affineness.

        Returns:
            bool: True if operation is affine-preserving.
        """
        pass

    @abstractmethod
    def _operation_name(self) -> str:
        """Name of the operation for tree representation.

        Returns:
            str: Operation name.
        """
        pass

    def tree(self) -> ExprTree:
        """Return tree representation for unary operation.

        Returns:
            ExprTree: Tree with operation name and operand child.
        """
        return ExprTree(self._operation_name(), self.operand.tree())


class ReduceSum(_Unary):
    """Sum reduction operation.

    Represents the sum of elements in an expression, optionally along a dimension.

    Args:
        operand: Expression to sum.
        dim: Dimension to sum over (None for all dimensions).
    """

    def __init__(self, operand: Expression, dim=None):
        """Initialize reduce sum operation.

        Args:
            operand: Expression or value to sum.
            dim: Dimension to sum over (None for all dimensions).
        """
        super().__init__(operand)
        self.dim = dim

    def _forward_op(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply sum operation.

        Args:
            tensor: Input tensor.

        Returns:
            torch.Tensor: Sum of tensor elements.
        """
        return torch.sum(tensor, dim=self.dim)

    def _is_affine_preserving(self) -> bool:
        """Sum is affine-preserving.

        Returns:
            bool: Always True.
        """
        return True

    def _operation_name(self) -> str:
        """Operation name for tree.

        Returns:
            str: 'sum' or 'sum_{dim}'.
        """
        return f"sum_{self.dim}" if self.dim is not None else "sum"


class Transpose(_Unary):
    """Matrix transpose operation.

    Represents the transpose of a 2D or higher-dimensional expression.
    Transposes the last two dimensions.

    Args:
        operand: Expression to transpose.
    """

    def _forward_op(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply transpose operation.

        Args:
            tensor: Input tensor.

        Returns:
            torch.Tensor: Transposed tensor.
        """
        return tensor.transpose(-2, -1)

    def _is_affine_preserving(self) -> bool:
        """Transpose is affine-preserving.

        Returns:
            bool: Always True.
        """
        return True

    def _operation_name(self) -> str:
        """Operation name for tree.

        Returns:
            str: 'transpose'.
        """
        return "transpose"


class Power(_Unary):
    """Element-wise power operation.

    Represents raising an expression to a power element-wise.

    Args:
        operand: Expression to raise to a power.
        exponent: Power to raise to (scalar).
    """

    def __init__(self, operand: Expression, exponent: float | int):
        """Initialize power operation.

        Args:
            operand: Expression to raise to a power.
            exponent: Power to raise to.
        """
        super().__init__(operand)
        self.exponent = exponent

    def _forward_op(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply power operation.

        Args:
            tensor: Input tensor.

        Returns:
            torch.Tensor: Tensor raised to exponent.
        """
        return torch.pow(tensor, self.exponent)

    def _is_affine_preserving(self) -> bool:
        """Power is affine-preserving only if exponent is 1.

        Returns:
            bool: True if exponent is 1.
        """
        return self.exponent == 1

    def _operation_name(self) -> str:
        """Operation name for tree.

        Returns:
            str: 'power_{exponent}'.
        """
        return f"power_{self.exponent}"
