"""OperatorSplit class for representing composite objective functions.

In optimization, a composite function is a function that consists of a smooth term and a
(typically non-smooth) proximal term.

The core functionality enables efficient splitting methods by providing access
to the smooth component, its gradient, and the proximal operator for the
non-smooth component. This is useful in first-order optimization algorithms such
as proximal gradient descent and its variants.
"""

from typing import Callable

import torch

from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict


class OperatorSplit:
    """Represents a composite objective function with a smooth and a proximal term.

    This class is designed for use in operator splitting methods such as
    proximal gradient descent. It provides access to the smooth part (`f`),
    the non-smooth/proximal part (`r`), their evaluations, the gradient of `f`,
    and the proximal operator of `r`.

    Args:
        smooth_expr (Expression): A differentiable (smooth) expression object.
        prox_expr (Expression, optional): A proxable (possibly non-smooth) expression
            object. If None, only the smooth term is used.

    Raises:
        ValueError: If `smooth_expr` is not smooth.
        ValueError: If `prox_expr` is provided and is not proxable.
        ValueError: If the parameter structures of `smooth_expr` and `prox_expr`
            are incompatible.

    Attributes:
        f (Expression): The smooth component of the composite function.
        r (Expression): The proximal (possibly non-smooth) component.
    """

    def __init__(self, smooth_expr: Expression, prox_expr: Expression | None = None):
        """Initializes the OperatorSplit object."""
        _validate_input(smooth_expr, prox_expr)
        self._f = smooth_expr
        self._r = prox_expr
        self._prox = _build_prox(self._r)

    @property
    def f(self) -> Expression:
        """Returns the smooth component of the composite function."""
        return self._f

    @property
    def r(self) -> Expression | None:
        """Returns the proximal component of the composite function."""
        return self._r

    def evaluate(self, variable_values: TensorDict) -> torch.Tensor:
        """Evaluate the composite objective function `f + r` at the given variables.

        Args:
            variable_values (TensorDict): A dictionary of variables.

        Returns:
            torch.Tensor: The scalar value of the objective function
                at `variable_values`.
        """
        f_eval = self._f.evaluate(variable_values)

        if self._r:
            r_eval = self._r.evaluate(variable_values)
            return f_eval + r_eval

        return f_eval

    def f_func(self, variable_values: TensorDict) -> torch.Tensor:
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
        return torch.func.grad(self.f_func)(variable_values)

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
        return self._prox(variable_values, eta)


def _validate_input(smooth_expr: Expression, prox_expr: Expression | None):
    if not smooth_expr.is_smooth():
        raise ValueError("Smooth expression is not smooth.")

    if prox_expr and not prox_expr.is_proxable():
        raise ValueError("Proximal expression is not proxable.")


def _build_prox(
    prox_expr: Expression | None,
) -> Callable[[TensorDict, float], TensorDict]:
    if prox_expr:
        if isinstance(prox_expr, AddExpression):
            num_non_smooth_exprs = prox_expr._num_non_smooth_exprs
        else:
            num_non_smooth_exprs = 1

        if num_non_smooth_exprs > 1:

            def prox(var_vals: TensorDict, eta: float) -> TensorDict:
                return prox_expr.prox(var_vals, eta)
        else:

            def prox(var_vals: TensorDict, eta: float) -> TensorDict:
                return var_vals.apply(lambda p: prox_expr.prox(p, eta))
    else:

        def prox(var_vals: TensorDict, eta: float) -> TensorDict:
            return var_vals

    return prox
