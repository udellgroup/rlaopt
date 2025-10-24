"""Utility functions for expression module."""

import torch

from rlaopt.expression.constant import ConstExpression
from rlaopt.expression.expression import Expression


def _to_expr(val) -> Expression:
    """Convert a value to an Expression if it isn't already.

    Args:
        val: Value to convert. Can be an Expression, float, int, or torch.Tensor.

    Returns:
        Expression: The converted expression.

    Raises:
        TypeError: If val cannot be converted to an Expression.

    """
    if isinstance(val, Expression):
        return val
    if isinstance(val, (float, int, torch.Tensor)):
        return ConstExpression(val)
    raise TypeError(f"Cannot convert {type(val)} to Expression")
