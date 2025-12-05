"""ProxGradSplit class for representing composite objective functions."""

import torch
from tensordict import merge_tensordicts

from rlaopt.atoms import Atom
from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.operator_split import _OperatorSplit


class ProxGradSplit(_OperatorSplit):
    """Represents a composite objective function with a smooth and a proximal term.

    This class is designed for use in proximal gradient descent.

    Args:
        expr (Expression): An expression object representing the composite function.

    Raises:
        ValueError: If the expression cannot be split into smooth and proxable parts.
    """

    def __init__(self, expr: Expression):
        """Initialize ProxGradSplit by validating and splitting the expression.

        Args:
            expr: Expression to split into smooth and proximal components

        Raises:
            ValueError: If expression cannot be split for proximal gradient
        """
        # Cast to AddExpression for easier splitting
        if not isinstance(expr, AddExpression):
            expr = AddExpression(expr)

        f, r = self._attempt_split(expr)
        super().__init__(f, r)

    def _attempt_split(self, expr: AddExpression) -> tuple[Expression, list[Atom]]:
        """Validate and split expression for proximal gradient methods.

        Args:
            expr: The expression to split

        Returns:
            tuple: (smooth_expr, proximal_atoms)

        Raises:
            ValueError: If expression cannot be split for proximal gradient
        """
        smooth_part = expr.get_smooth_part()
        non_smooth_exprs = expr.get_non_smooth_exprs()

        # All non-smooth terms must be proxable atoms
        if any(
            not (isinstance(term, Atom) and term.is_proxable())
            for term in non_smooth_exprs
        ):
            raise ValueError(
                "All non-smooth terms must be proxable atoms for ProxGradSplit."
            )

        # All non-smooth terms must be proxable atoms
        if any(
            not (isinstance(term, Atom) and term.is_proxable())
            for term in non_smooth_exprs
        ):
            raise ValueError(
                "All non-smooth terms must be proxable atoms for ProxGradSplit."
            )

        # Check for variable disjointness (this is essential for proximal gradient)
        seen_variables = set()
        for term in non_smooth_exprs:
            expr_var_names = set(term.get_variable_names())
            if seen_variables & expr_var_names:
                raise ValueError(
                    "Non-smooth terms must operate on disjoint sets of variables."
                )
            seen_variables.update(expr_var_names)

            return smooth_part, non_smooth_exprs

    @property
    def variable_values(self) -> TensorDict:
        """Returns the variable values associated with the composite function."""
        td_f = self._f.variable_values
        tds_r = [r.variable_values for r in self._r]
        if not tds_r:
            return td_f
        return merge_tensordicts(td_f, *tds_r)

    def evaluate(self, variable_values: TensorDict) -> torch.Tensor:
        """Evaluate the composite objective function `f + r` at the given variables.

        Args:
            variable_values (TensorDict): A dictionary of variables.

        Returns:
            torch.Tensor: The scalar value of the objective function
                at `variable_values`.
        """
        val_f = self._f.evaluate(variable_values)
        val_r = sum(r.evaluate(variable_values) for r in self._r)
        return val_f + val_r
