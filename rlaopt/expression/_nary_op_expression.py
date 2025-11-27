from abc import ABC, abstractmethod

import torch

from rlaopt.expression.expression import Expression
from rlaopt.expression.tree import ExprTree


class _NAryOpExpression(Expression, ABC):
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

    def __init__(self, *exprs: Expression):
        """Initialize with variable number of expressions.

        Args:
            *exprs: Expressions to combine. Last argument can be None.

        Raises:
            ValueError: If no expressions provided.
        """
        super().__init__()

        if not exprs:
            raise ValueError(f"{self.__class__.__name__} requires at least one operand")
        self.exprs = torch.nn.ModuleList(exprs)

    @abstractmethod
    def op(self, values: list[torch.Tensor]):
        """Apply the operator to evaluated tensors.

        Args:
            values: List of evaluated tensor expressions.

        Returns:
            torch.Tensor: Result of the operation.
        """
        pass

    @abstractmethod
    def is_commutative_operation(self) -> bool:
        """Check if this operation is commutative.

        Subclasses must implement this to indicate whether the operation
        is commutative (e.g., addition, multiplication) or not.

        Returns:
            bool: True if the operation is commutative, False otherwise.
        """
        pass

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
        return len(self.exprs)

    def tree(self) -> ExprTree:
        """Return tree representation for n-ary operation.

        Returns:
            ExprTree: Tree with operation type and child trees.
        """
        return ExprTree(
            self.__class__.__name__,
            *(expr.tree() for expr in self.exprs),
            is_commutative=self.is_commutative_operation(),
        )
