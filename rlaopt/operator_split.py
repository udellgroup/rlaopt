"""OperatorSplit class for representing composite objective functions.

In optimization, a composite function is a function that consists of a smooth term and a
(typically non-smooth) proximal term.

The core functionality enables efficient splitting methods by providing access
to the smooth component, its gradient, and the proximal operator for the
non-smooth component. This is useful in first-order optimization algorithms such
as proximal gradient descent and its variants.
"""

import torch
from tensordict import merge_tensordicts

from rlaopt.atoms import Atom
from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict


class OperatorSplit:
    """Represents a composite objective function with a smooth and a proximal term.

    This class is designed for use in operator splitting methods such as
    proximal gradient descent. It provides access to the smooth part (`f`),
    the non-smooth/proximal part (`r`), their evaluations, the gradient of `f`,
    and the proximal operator of `r`.

    Args:
        expr (Expression): An expression object representing the composite function.

    Raises:
        ValueError: If the expression cannot be split into smooth and proxable parts.
    """

    def __init__(self, expr: Expression):
        """Initializes the OperatorSplit object."""
        # Start by casting to AddExpression for easier splitting
        if not isinstance(expr, AddExpression):
            expr = AddExpression(expr)

        self._f, self._r = _attempt_split(expr)

    @property
    def f(self) -> Expression:
        """Returns the smooth component of the composite function."""
        return self._f

    @property
    def r(self) -> list[Atom]:
        """Returns the proximal component of the composite function."""
        return self._r

    @property
    def variable_values(self) -> TensorDict:
        """Returns the variable values associated with the composite function."""
        td_f = self._f.variable_values
        tds_r = [r.variable_values for r in self._r]
        if not tds_r:
            return td_f
        return merge_tensordicts(td_f, *tds_r)

    @property
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

    def func_f(self, variable_values: TensorDict) -> torch.Tensor:
        """Evaluate the smooth part of the objective function at the given variables.

        Args:
            variable_values (TensorDict): A dictionary of variables.

        Returns:
            torch.Tensor: The scalar value of the smooth part at `variable_values`.
        """
        return self._f.evaluate(variable_values)

    def grad_f(self, variable_values: TensorDict) -> TensorDict:
        """Compute the gradient of the smooth part of the objective function.

        Args:
            variable_values (TensorDict): A dictionary of variables.

        Returns:
            TensorDict: A dictionary of gradients with the same structure
                as `variable_values`.
        """
        return torch.func.grad(self.func_f)(variable_values)

    def hvp_f(self, variable_values: TensorDict, v: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian-vector product of the smooth part of the objective.

        Args:
            variable_values (TensorDict): A dictionary of variables.
            v (torch.Tensor): A torch tensor of shape (variable_values.dim,)
                representing the vector to multiply with the Hessian.

        Returns:
            torch.Tensor: The Hessian-vector product of the Hessian at
                variable_values and v.
        """

        def g_dot_v(var_vals: TensorDict) -> torch.Tensor:
            return torch.dot(self.grad_f(var_vals).to_flat_tensor(), v)

        return (torch.func.grad(g_dot_v)(variable_values)).to_flat_tensor()

    def prox(self, variable_values: TensorDict, eta: float) -> TensorDict:
        """Apply the proximal operator of `r` with step size `eta` to the variables.

        Args:
            variable_values (TensorDict): A dictionary of variables.
            eta (float): Step size or scaling factor for the proximal operator.

        Returns:
            TensorDict: Updated variables after applying the proximal operator.
        """
        for r in self._r:
            variable_values_update = r.prox(variable_values, eta)
            variable_values.update(variable_values_update)
        return variable_values


def _attempt_split(expr: AddExpression) -> tuple[Expression, list[Atom]]:
    smooth_part = expr.get_smooth_part()
    non_smooth_exprs = expr.get_non_smooth_exprs()

    # All non-smooth terms must be proxable atoms
    if any(
        not (isinstance(expr, Atom) and expr.is_proxable()) for expr in non_smooth_exprs
    ):
        raise ValueError(
            "All non-smooth terms must be proxable atoms for OperatorSplit."
        )

    # Check for variable disjointness (this is essential for proximal gradient)
    seen_variables = set()
    for expr in non_smooth_exprs:
        expr_var_names = set(expr.get_variable_names())
        if seen_variables & expr_var_names:
            raise ValueError(
                "Non-smooth terms must operate on disjoint sets of variables."
            )
        seen_variables.update(expr_var_names)

    return smooth_part, non_smooth_exprs
