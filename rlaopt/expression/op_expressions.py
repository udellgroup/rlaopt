"""Module for addition and product operations on expressions."""

from typing import Callable

import torch

from rlaopt.expression import expr_types
from rlaopt.expression._nary_op_expression import _NAryOpExpression
from rlaopt.expression.constant import ConstExpression
from rlaopt.expression.expression import Expression
from rlaopt.ext_tensordict import TensorDict


class AddExpression(_NAryOpExpression):
    """Sum of multiple expressions.

    Represents the sum of two or more expressions.

    Args:
        *exprs: Expressions to sum together.
    """

    def __init__(self, *exprs):
        """Initialize sum expression.

        Note: Flattening and optimization are handled by _create_add in utils.py.
        This constructor assumes it receives already-optimized expressions.

        Args:
            *exprs: Variable number of expressions to sum.
        """
        super().__init__(*exprs)
        self._prox = self._build_prox()

        # Build op method for adding expressions
        self._op = self._build_op()

    def op(self, values: list[torch.Tensor]) -> torch.Tensor:
        """Apply the addition operator to evaluated expressions.

        Args:
            values: List of evaluated tensor expressions.

        Returns:
            torch.Tensor: Sum of all expressions.

        Raises:
            ValueError: If expression list is empty.
        """
        return self._op(values)

    def is_proxable(self):
        """Check if sum is proxable.

        A sum is proxable if:
        1. All non-smooth terms are proxable, AND
        2. Non-smooth terms operate on disjoint sets of variables.

        Returns:
            bool: True if the sum is proxable.
        """
        non_smooth_exprs = self._get_non_smooth_exprs()

        # All non-smooth terms must be proxable
        if any(not expr.is_proxable() for expr in non_smooth_exprs):
            return False

        # Check for variable overlap
        seen_variables = set()
        for expr in non_smooth_exprs:
            expr_var_names = set(expr.get_variable_names())
            if seen_variables & expr_var_names:  # intersection
                return False
            seen_variables.update(expr_var_names)

        return True

    def prox(self, location, prox_scaling):
        """Compute proximal operator of the sum.

        Args:
            location: Point at which to evaluate proximal operator.
            prox_scaling: Scaling factor for proximal operator.

        Returns:
            torch.Tensor or dict: Result of proximal operator.

        Raises:
            NotImplementedError: If the sum is not proxable.
        """
        return self._prox(location, prox_scaling)

    def _get_smooth_exprs(self) -> list[Expression]:
        return [e for e in self.exprs if e.is_smooth()]

    def _get_non_smooth_exprs(self) -> list[Expression]:
        return [e for e in self.exprs if not e.is_smooth()]

    def get_smooth_part(self) -> Expression | None:
        """Get the smooth part of the sum expression.

        Returns:
            Expression or None: Smooth part of the sum, or None if no smooth terms.
        """
        smooth_exprs = self._get_smooth_exprs()
        if not smooth_exprs:
            return None
        return AddExpression(*smooth_exprs)

    def get_non_smooth_part(self) -> Expression | None:
        """Get the non-smooth part of the sum expression.

        Returns:
            Expression or None: Non-smooth part of the sum, or None if
                no non-smooth terms.
        """
        non_smooth_exprs = self._get_non_smooth_exprs()
        if not non_smooth_exprs:
            return None
        return AddExpression(*non_smooth_exprs)

    @property
    def num_non_smooth_exprs(self):
        """Get count of non-smooth terms.

        Returns:
            int: Number of non-smooth expressions in the sum.
        """
        return len(self._get_non_smooth_exprs())

    def _build_op(self) -> Callable[[list[torch.Tensor]], torch.Tensor]:
        """Build the addition op method for AddExpression."""
        if self.n_exprs > 1:

            def add_op(values: list[torch.Tensor]) -> torch.Tensor:
                result = values[0]
                for value in values[1:]:
                    result = result + value
                return result
        else:

            def add_op(values: list[torch.Tensor]) -> torch.Tensor:
                return values[0]

        return add_op

    def _build_prox(self):
        """Build the proximal operator function.

        Returns:
            Callable: Proximal operator function.
        """
        if not self.is_proxable():

            def prox(location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
                raise NotImplementedError("Expression is not proxable")

            return prox

        elif self.num_non_smooth_exprs == 1:
            proxes = self._get_proxes()
            return proxes[0]

        else:
            proxes = self._get_proxes()

            def prox(location: TensorDict, prox_scaling: float) -> TensorDict:
                return TensorDict(
                    {
                        loc_name: prox_fn(loc, prox_scaling)
                        for (loc_name, loc), prox_fn in zip(location.items(), proxes)
                    }
                )

            return prox

    def _get_proxes(self) -> list[Callable[[torch.Tensor, float], torch.Tensor]]:
        """Get proximal operators of non-smooth expressions.

        Returns:
            list: List of proximal operator functions.
        """
        proxes = []
        for expr in self.exprs:
            if not expr.is_smooth():
                proxes.append(expr.prox)
        return proxes

    def tree(self) -> tuple:
        """Return tree representation for AddExpression.

        Returns:
            tuple: ('AddExpression', child1_tree, child2_tree, ...)
        """
        return ("AddExpression",) + tuple(expr.tree() for expr in self.exprs)


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

    def __init__(self, *exprs, matmul: bool = False):
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
        if isinstance(expr, (expr_types.variable(), ConstExpression)):
            return True
        if isinstance(expr, (ProductExpression, AddExpression)):
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

    def is_proxable(self):
        """Products are not proxable.

        Returns:
            bool: Always False.
        """
        return False

    def prox(self, location, prox_scaling):
        """Products don't have proximal operators.

        Args:
            location: Unused.
            prox_scaling: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ProductExpression is not proxable")

    def tree(self) -> tuple:
        """Return tree representation for ProductExpression.

        Returns:
            tuple: ('ProductExpression', child1_tree, child2_tree, ...)
        """
        return ("ProductExpression",) + tuple(expr.tree() for expr in self.exprs)

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
