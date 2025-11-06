"""Module for addition and product operations on expressions."""

from typing import Callable

import torch

from rlaopt.expression import expr_types
from rlaopt.expression._nary_op_expression import _NAryOpExpression
from rlaopt.expression.constant import ConstExpression
from rlaopt.ext_tensordict import TensorDict


class AddExpression(_NAryOpExpression):
    """Sum of multiple expressions.

    Represents the sum of two or more expressions.

    Args:
        *exprs: Expressions to sum together.

    Examples:
        >>> x = Variable((5,))
        >>> y = Variable((5,))
        >>> z = x + y + 10
        >>> isinstance(z, AddExpression)
        True
    """

    def __init__(self, *exprs):
        """Initialize sum expression with flattening.

        Automatically flattens nested AddExpressions to prevent
        deeply nested structures.

        Args:
            *exprs: Variable number of expressions to sum.
        """
        # Flatten nested AddExpressions before calling super().__init__
        # This ensures that (a + b) + c becomes AddExpression(a, b, c)
        # rather than AddExpression(AddExpression(a, b), c)
        flattened_exprs = []

        # Flatten any nested AddExpressions
        for expr in exprs:
            if isinstance(expr, AddExpression):
                # Recursively flatten nested AddExpressions
                flattened_exprs.extend(expr.exprs)
            else:
                flattened_exprs.append(expr)

        # Call parent constructor with flattened expressions
        super().__init__(*flattened_exprs)
        self._num_non_smooth_exprs = self._count_non_smooth_terms()
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
        non_smooth_exprs = [e for e in self.exprs if not e.is_smooth()]

        # All non-smooth terms must be proxable
        if any(not expr.is_proxable() for expr in non_smooth_exprs):
            return False

        # Check for variable overlap
        seen_variables = set()
        for expr in non_smooth_exprs:
            expr_var_names = expr.get_variable_names()
            if seen_variables & expr_var_names:  # intersection
                return False
            seen_variables.update(expr_var_names)

        return True

    def operator_split(self):
        """Split sum into smooth and non-smooth parts.

        Separates the sum into two expressions: one containing all smooth terms
        and one containing all non-smooth terms. Useful for proximal algorithms.

        Returns:
            tuple: (smooth_expr, non_smooth_expr) where either can be None.

        Examples:
            >>> from rlaopt.atoms import L1Norm, SumSquares
            >>> x = Variable((5,))
            >>> expr = SumSquares(x) + L1Norm(x)
            >>> smooth, non_smooth = expr.operator_split()
            >>> smooth.is_smooth()
            True
            >>> non_smooth.is_smooth()
            False
        """
        smooth = [e for e in self.exprs if e.is_smooth()]
        non_smooth = [e for e in self.exprs if not e.is_smooth()]
        smooth_expr = AddExpression(*smooth) if smooth else None
        non_smooth_expr = AddExpression(*non_smooth) if non_smooth else None

        return smooth_expr, non_smooth_expr

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

    @property
    def num_non_smooth_exprs(self):
        """Get count of non-smooth terms.

        Returns:
            int: Number of non-smooth expressions in the sum.
        """
        return self._num_non_smooth_exprs

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

        # Always return a function that handles TensorDict
        proxes = self._get_proxes()

        def prox(location: TensorDict, prox_scaling: float) -> TensorDict:
            return TensorDict(
                {
                    loc_name: prox_fn(loc, prox_scaling)
                    for (loc_name, loc), prox_fn in zip(location.items(), proxes)
                }
            )

        return prox

    def _count_non_smooth_terms(self):
        """Count non-smooth expressions in the sum.

        Returns:
            int: Number of non-smooth terms.
        """
        return sum(1 for expr in self.exprs if not expr.is_smooth())

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

    Examples:
        >>> x = Variable((5,))
        >>> y = Variable((5,))
        >>> z = x * y  # Elementwise
        >>> A = Variable((3, 4))
        >>> b = Variable((4,))
        >>> c = A @ b  # Matrix multiplication
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
