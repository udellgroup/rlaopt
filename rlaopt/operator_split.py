"""
This module defines the OperatorSplit class for representing composite objective
functions in optimization problems that consist of a smooth term and a
(non-smooth) proximal term.

The core functionality enables efficient splitting methods by providing access
to the smooth component, its gradient, and the proximal operator for the
non-smooth component. This is useful in first-order optimization algorithms such
as Proximal Gradient Descent and its variants.

Classes:
    OperatorSplit: Encapsulates a smooth and a proximal expression for
                   operator splitting in optimization.

Functions:
    _validate_input: Internal helper to validate the input expressions.
    _build_prox: Internal helper to construct the correct proximal operator.
"""

from typing import Callable

import torch

from rlaopt.expression.expression import Expression, AddExpression
from rlaopt.utils.tensor_dict_ops import dict_map, flatten
from rlaopt._typing import TensorDict

class OperatorSplit:
    """
    Represents a composite objective function with a smooth and a proximal term.

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

    def __init__(self, smooth_expr: Expression, prox_expr: Expression = None):
        _validate_input(smooth_expr, prox_expr)
        self._f = smooth_expr
        self._r = prox_expr
        self._prox = _build_prox(self._r)

    @property
    def f(self) -> Expression:
        return self._f
    
    @property
    def r(self) -> Expression:
        return self._r
    
    def evaluate(self, params: TensorDict) -> torch.Tensor:
        """
        Evaluate the full composite objective function `f + r` at the given parameters.

        Args:
            params (TensorDict): A dictionary of parameters compatible with f.

        Returns:
            torch.Tensor: The scalar value of the objective function at `params`.
        """
        return self._f.evaluate(params) + self._r.evaluate(self._r.expr_convert_params(params))
    
    def f_func(self, params: TensorDict) -> torch.Tensor:
        """
        Evaluate the smooth part of the objective function at the given parameters.

        Args:
            params (TensorDict): A dictionary of parameters.

        Returns:
            torch.Tensor: The scalar value of the smooth part at `params`.
        """
        return self._f.evaluate(params)
    
    def grad_f(self, params: TensorDict) -> TensorDict:
        """
        Compute the gradient of the smooth part of the objective function.

        Args:
            params (TensorDict): A dictionary of parameters.

        Returns:
            TensorDict: A dictionary of gradients with the same structure as `params`.
        """
        return torch.func.grad(self.f_func)(params)
    
    def hvp_f(self, params: TensorDict, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian-vector product of the smooth part of the objective function.

        Args:
            params (TensorDict): A dictionary of parameters.
            v torch.Tensor: A torch tensor of shape (params.dim,) representing the vector to multiply with the Hessian.

        Returns:
            torch.Tensor: The Hessian-vector product of the Hessian at params and v.
        """
        def g_dot_v(params: TensorDict) -> torch.Tensor:
            return torch.dot(flatten(self.grad_f(params)), v)
        
        return flatten(torch.func.grad(g_dot_v)(params))
    
    def prox(self, params: TensorDict, eta: float) -> TensorDict:
        """
        Apply the proximal operator of `r` with step size `eta` to the parameters.

        Args:
            params (TensorDict): A dictionary of parameters.
            eta (float): Step size or scaling factor for the proximal operator.

        Returns:
            TensorDict: Updated parameters after applying the proximal operator.
        """
        return self._prox(params, eta)
    
def _validate_input(smooth_expr: Expression, prox_expr: Expression)-> None:
    if not smooth_expr.is_smooth():
            raise ValueError("Smooth expression is not smooth.")
        
    if prox_expr and not prox_expr.is_proxable():
            raise ValueError("Proximal expression is not proxable.")


def _build_prox(prox_expr: Expression) -> Callable[[TensorDict, float], TensorDict]:
     if prox_expr:
            if isinstance(prox_expr, AddExpression):
                num_non_smooth_exprs = prox_expr._num_non_smooth_exprs
            else:
                num_non_smooth_exprs = 1
            
            if num_non_smooth_exprs > 1:
               def prox(
                    params: TensorDict, eta: float
                ) -> TensorDict:
                    return prox_expr.prox(params, eta)
            else: 
                 def prox(params: TensorDict, eta: float) -> TensorDict:
                    return dict_map(params, lambda p: prox_expr.prox(p, eta))
     else:
                 def prox(params: TensorDict, eta: float) -> TensorDict:
                    return params
     return prox
        