"""OperatorSplit class for representing composite objective functions.

In optimization, a composite function is a function that consists of a smooth term and a
(typically non-smooth) proximal term.

The core functionality enables efficient splitting methods by providing access
to the smooth component, its gradient, and the proximal operator for the
non-smooth component. This is useful in first-order optimization algorithms such
as proximal gradient descent and its variants.

Functions:
    _validate_input: Internal helper to validate the input expressions.
    _build_prox: Internal helper to construct the correct proximal operator.
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

    def evaluate(self, variables_dict: TensorDict) -> torch.Tensor:
        """Evaluate the composite objective function `f + r` at the given variables.

        Args:
            variables_dict (TensorDict): A dictionary of variables compatible with f.

        Returns:
            torch.Tensor: The scalar value of the objective function
                at `variables_dict`.
        """
        f_var_names = self._f.get_variable_names()
        f_vars = variables_dict.select(*f_var_names)
        f_eval = self._f.evaluate(f_vars)

        if self._r:
            r_var_names = self._r.get_variable_names()
            r_vars = variables_dict.select(*r_var_names)
            r_eval = self._r.evaluate(r_vars)
            return f_eval + r_eval

        return f_eval

    def f_func(self, variables_dict: TensorDict) -> torch.Tensor:
        """Evaluate the smooth part of the objective function at the given variables.

        Args:
            variables_dict (TensorDict): A dictionary of variables.

        Returns:
            torch.Tensor: The scalar value of the smooth part at `variables_dict`.
        """
        f_var_names = self._f.get_variable_names()
        params = variables_dict.select(*f_var_names)
        return self._f.evaluate(params)

    def grad_f(self, variables_dict: TensorDict) -> TensorDict:
        """Compute the gradient of the smooth part of the objective function.

        Args:
            variables_dict (TensorDict): A dictionary of variables.

        Returns:
            TensorDict: A dictionary of gradients with the same structure
                as `variables_dict`.
        """
        return torch.func.grad(self.f_func)(variables_dict)

    def hvp_f(self, variables_dict: TensorDict, v: torch.Tensor) -> torch.Tensor:
        """Compute the Hessian-vector product of the smooth part of the objective.

        Args:
            variables_dict (TensorDict): A dictionary of variables.
            v (torch.Tensor): A torch tensor of shape (variables_dict.dim,) representing
                the vector to multiply with the Hessian.

        Returns:
            torch.Tensor: The Hessian-vector product of the Hessian at
                variables_dict and v.
        """

        def g_dot_v(params: TensorDict) -> torch.Tensor:
            return torch.dot(self.grad_f(params).to_flat_tensor(), v)

        return (torch.func.grad(g_dot_v)(variables_dict)).to_flat_tensor()

    def prox(self, variables_dict: TensorDict, eta: float) -> TensorDict:
        """Apply the proximal operator of `r` with step size `eta` to the variables.

        Args:
            variables_dict (TensorDict): A dictionary of variables.
            eta (float): Step size or scaling factor for the proximal operator.

        Returns:
            TensorDict: Updated variables after applying the proximal operator.
        """
        return self._prox(variables_dict, eta)


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

            def prox(params: TensorDict, eta: float) -> TensorDict:
                return prox_expr.prox(params, eta)
        else:

            def prox(params: TensorDict, eta: float) -> TensorDict:
                return params.apply(lambda p: prox_expr.prox(p, eta))
    else:

        def prox(params: TensorDict, eta: float) -> TensorDict:
            return params

    return prox
