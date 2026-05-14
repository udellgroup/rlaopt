"""Module for sum and product operations on expressions."""

from typing import Callable

import torch

from rlaopt.expression import expr_types
from rlaopt.expression._nary_op_expression import _NAryOpExpression
from rlaopt.expression.constant import Constant
from rlaopt.expression.expression import Expression


class SumExpression(_NAryOpExpression):
    """Sum of multiple expressions.

    Represents the sum of two or more expressions.

    Args:
        *exprs: Expressions to sum together.
    """

    def __init__(self, *exprs: Expression):
        """Initialize sum expression.

        Note: Flattening and optimization are handled by _create_add in utils.py.
        This constructor assumes it receives already-optimized expressions.

        Args:
            *exprs: Variable number of expressions to sum.
        """
        super().__init__(*exprs)

        # Build op method for summing expressions
        self._op = self._build_op()

    def op(self, values: list[torch.Tensor]) -> torch.Tensor:
        """Apply the sum operator to evaluated expressions.

        Args:
            values: List of evaluated tensor expressions.

        Returns:
            torch.Tensor: Sum of all expressions.

        Raises:
            ValueError: If expression list is empty.
        """
        return self._op(values)

    def is_commutative_operation(self) -> bool:
        """Addition is commutative.

        Returns:
            bool: Always True.
        """
        return True

    def is_affine(self) -> bool:
        """Check if the sum expression is affine.

        Returns:
            bool: True if all summed expressions are affine, False otherwise.
        """
        return all(e.is_affine() for e in self.exprs)

    def get_non_smooth_exprs(self) -> list[Expression]:
        """Get the non-smooth part of the sum expression as a list of expressions.

        Returns:
            list[Expression]: Non-smooth part of the sum.
        """
        return [e for e in self.exprs if not e.is_smooth()]

    def get_smooth_part(
        self, return_mode: str = "expression"
    ) -> Expression | list[Expression]:
        """Get the smooth part of the sum expression.

        Returns:
            Expression: Smooth part of the sum.
        """
        smooth_exprs = [e for e in self.exprs if e.is_smooth()]

        if not smooth_exprs:
            return Constant(0.0)

        if return_mode == "list":
            return smooth_exprs

        return SumExpression(*smooth_exprs)

    def _build_op(self) -> Callable[[list[torch.Tensor]], torch.Tensor]:
        """Build the sum op method for SumExpression."""
        if self.n_exprs > 1:

            def sum_op(values: list[torch.Tensor]) -> torch.Tensor:
                result = values[0]
                for value in values[1:]:
                    result = result + value
                return result
        else:

            def sum_op(values: list[torch.Tensor]) -> torch.Tensor:
                return values[0]

        return sum_op


class ProductExpression(_NAryOpExpression):
    """Product of multiple expressions.

    Represents either elementwise multiplication (*) or matrix multiplication (@)
    of multiple expressions. Validates that only simple expressions (Variables
    and Constants) are multiplied together.

    Args:
        *exprs: Expressions to multiply.
        matmul: If True, use matrix multiplication; if False, elementwise.

    Attributes:
        matmul: Whether to use matrix multiplication.
    """

    def __init__(self, *exprs: Expression, matmul: bool = False):
        """Initialize product expression.

        Args:
            *exprs: Expressions to multiply.
            matmul: Whether to use matrix multiplication.

        Raises:
            TypeError: If multiplying complex parameterized expressions.
        """
        self.matmul = matmul
        super().__init__(*exprs)

        # Valid input
        self._validate()

        # Build product op
        self._op = self._build_op()

    def _validate(self):
        """Validate that multiplication is allowed.

        Only allows multiplication of Variables and Constants, or expressions
        built purely from Variables and Constants using basic operations.

        Raises:
            TypeError: If validation fails.
        """
        var_exprs = [e for e in self.exprs if e.get_variable_names()]

        if len(var_exprs) > 1:
            if not all(self._is_var_or_const_tree(e) for e in var_exprs):
                raise TypeError(
                    "Cannot multiply two arbitrary parameterized Expressions. "
                    "Only Variables and Constants can be multiplied together."
                )

    def _is_var_or_const_tree(self, expr) -> bool:
        """Check if expression tree contains only Variables and Constants.

        Args:
            expr: Expression to check.

        Returns:
            bool: True if tree contains only simple expressions.
        """
        if isinstance(expr, (expr_types.variable(), Constant)):
            return True
        if isinstance(expr, (ProductExpression, SumExpression)):
            return all(self._is_var_or_const_tree(child) for child in expr.exprs)
        if isinstance(expr, expr_types.unary_op_expr()):
            return self._is_var_or_const_tree(expr.operand)
        return False

    def op(self, values):
        """Apply multiplication operator.

        Args:
            values: List of evaluated tensor expressions.

        Returns:
            torch.Tensor or cp.Expression: Product of all expressions.

        Raises:
            ValueError: If expression list is empty.
        """
        return self._op(values)

    def is_commutative_operation(self) -> bool:
        """Multiplication is commutative except for matrix multiplication.

        Returns:
            bool: True if elementwise, False if matrix multiplication.
        """
        return not self.matmul

    def is_affine(self) -> bool:
        """Check if the product expression is affine.

        Returns:
            bool: True if at most one expression is non-constant and
                that expression is affine.
        """
        if all(isinstance(e, Constant) for e in self.exprs):
            return True
        non_const_exprs = [e for e in self.exprs if not isinstance(e, Constant)]
        if len(non_const_exprs) == 1:
            return non_const_exprs[0].is_affine()
        return False

    def _build_op(self) -> Callable[[list[torch.Tensor]], torch.Tensor]:
        if self.matmul:

            def prod_vals(values: list[torch.Tensor]) -> torch.Tensor:
                res = values[0]
                for value in values[1:]:
                    res = torch.matmul(res, value)
                return res
        else:

            def prod_vals(values: list[torch.Tensor]) -> torch.Tensor:
                res = values[0]
                for value in values[1:]:
                    res = res * value
                return res

        if self.n_exprs > 1:

            def prod_op(values: list[torch.Tensor]):
                return prod_vals(values)
        else:

            def prod_op(values: list[torch.Tensor]):
                return values[0]

        return prod_op
