from typing import Callable

import torch

from rlaopt.expression import Expression
from rlaopt.expression.tree import ExprTree
from rlaopt.expression.utils import _to_expr


class _UnaryOpExpression(Expression):
    """Unary operation applied to an expression.

    Represents the application of a unary function (e.g., abs, exp, sum)
    to another expression. The operation is stored as a lambda function.

    Args:
        operand: Expression to apply operation to.
        op: Unary function to apply (torch.Tensor -> torch.Tensor).

    Attributes:
        operand: The sub-expression.
        _op: The unary operation function.

    Examples:
        >>> x = Variable((5,))
        >>> x_squared = x ** 2  # UnaryOpExpression with torch.pow
        >>> x_sum = x.sum()  # UnaryOpExpression with torch.sum
    """

    def __init__(
        self,
        operand: Expression | float | int | torch.Tensor,
        op: Callable[[torch.Tensor], torch.Tensor],
        name: str | None = None,
    ):
        """Initialize unary operation expression.

        Args:
            operand: Expression or value to operate on.
            op: Function to apply.
            name: Optional name for the operation (e.g., 'sum', 'transpose').
                If not provided, uses the class name.
        """
        super().__init__()
        self.operand = _to_expr(operand)
        self.add_module("_operand", self.operand)
        self._op = op
        self._op_name = name

    def forward(self) -> torch.Tensor:
        """Evaluate the unary operation.

        Returns:
            torch.Tensor: Result of applying operation to operand.
        """
        val = self.operand.forward()
        return self._op(val)

    def is_smooth(self) -> bool:
        """Smoothness depends on the operand.

        Returns:
            bool: True if operand is smooth.
        """
        return self.operand.is_smooth()

    def is_proxable(self) -> bool:
        """Unary operations are generally not proxable.

        Returns:
            bool: Always False.
        """
        return False

    def tree(self) -> ExprTree:
        """Return tree representation for unary operation.

        Unary operations are not commutative since they have a single operand.

        Returns:
            ExprTree: Tree with operation name and operand child.
        """
        node_name = self._op_name if self._op_name else self.__class__.__name__
        return ExprTree(node_name, self.operand.tree())
