"""Utility functions for solvers."""

from rlaopt.expression.expression import AddExpression, Expression
from rlaopt.operator_split import OperatorSplit


def split_objective(obj: Expression) -> OperatorSplit:
    """Split objective into a smooth and proxable part."""
    if isinstance(obj, AddExpression):
        f, r = obj.operator_split()
        return OperatorSplit(f, r)
    else:
        return OperatorSplit(obj)
