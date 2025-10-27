from abc import ABC, abstractmethod

import torch

from rlaopt.expression.expression import Expression
from rlaopt.expression.utils import _to_expr


class _NAryOperatorExpression(Expression, ABC):
    """Private base class for n-ary operations (operations on multiple expressions).

    Provides infrastructure for expressions that operate on multiple sub-expressions,
    such as addition and multiplication. Handles parameter tracking and evaluation
    automatically.

    Args:
        *exprs: Variable number of expressions to operate on.

    Attributes:
        exprs: ModuleList of sub-expressions.
        _expr_signatures: Cached function signatures for kwargs filtering.
    """

    @abstractmethod
    def op(self, values: list[torch.Tensor]):
        """Apply the operator to evaluated tensors.

        Args:
            values: List of evaluated tensor expressions.

        Returns:
            torch.Tensor: Result of the operation.
        """
        pass

    def __init__(self, *exprs):
        """Initialize with variable number of expressions.

        Args:
            *exprs: Expressions to combine. Last argument can be None.

        Raises:
            ValueError: If no expressions provided.
        """
        super().__init__()

        if len(exprs) == 0:
            raise ValueError(f"{self.__class__.__name__} requires at least one operand")
        else:
            if exprs[-1] is not None:
                self.exprs = torch.nn.ModuleList([_to_expr(e) for e in exprs])
            else:
                # If last expr is None, ignore it
                # This is needed for operator_split in AddExpression
                self.exprs = torch.nn.ModuleList([_to_expr(e) for e in exprs[:-1]])
        self._n_exprs = len(self.exprs)

    def is_smooth(self) -> bool:
        """Check if all sub-expressions are smooth.

        Returns:
            bool: True if all sub-expressions are smooth.
        """
        return all(expr.is_smooth() for expr in self.exprs)

    def forward(self) -> torch.Tensor:
        """Evaluate the operation with current parameters.

        Evaluates each sub-expression and applies the operation.

        Returns:
            torch.Tensor: Result of the operation.
        """
        vals = [expr.forward() for expr in self.exprs]
        return self.op(vals)

    @property
    def n_exprs(self):
        """The number of expressions being operated on."""
        return self._n_exprs
