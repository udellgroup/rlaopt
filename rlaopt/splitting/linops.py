"""Helper linear operators for splitting representations."""

import torch
from linops import LinearOperator

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict


class _AffineExprLinOp(LinearOperator):
    """Linear operator for affine expressions of the form Ax - b."""

    def __init__(
        self, affine_expr: Expression, bias: torch.Tensor, smooth_expr_vars: TensorDict
    ):
        super().__init__()
        self._affine_expr = affine_expr
        self._bias = bias
        self._smooth_expr_vars = smooth_expr_vars
        self._unflatten = lambda v: smooth_expr_vars.from_flat_tensor(v)

        input_dim = smooth_expr_vars.flat_dim()
        self._shape = (self._bias.shape[0], input_dim)
        self._device = self._bias.device

    def _matmul_impl(self, v: torch.Tensor):
        v_td = self._unflatten(v)
        return self._affine_expr.evaluate(v_td) - self._bias

    def create_adjoint(self) -> LinearOperator:
        """Create the adjoint operator A.T explicitly."""
        return _AffineExprAdjointLinOp(
            self._affine_expr, self._smooth_expr_vars, self._shape
        )


class _AffineExprAdjointLinOp(LinearOperator):
    """Adjoint (transpose) of _AffineExprLinOp."""

    def __init__(
        self,
        affine_expr: Expression,
        smooth_expr_vars: TensorDict,
        forward_shape: tuple[int, int],
    ):
        super().__init__()
        self._affine_expr = affine_expr
        self._smooth_expr_vars = smooth_expr_vars

        # Transpose shape
        m, n = forward_shape
        self._shape = (n, m)
        self._device = smooth_expr_vars.to_flat_tensor().device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute A.T @ v using VJP."""
        zero_vars = self._smooth_expr_vars.apply(torch.zeros_like)

        def forward_fn(var_vals: TensorDict) -> torch.Tensor:
            return self._affine_expr.evaluate(var_vals)

        _, vjp_fn = torch.func.vjp(forward_fn, zero_vars)
        vjp_result = vjp_fn(v)[0]

        return vjp_result.to_flat_tensor()


class _HVPLinOp(LinearOperator):
    """Linear operator for Hessian-vector products of a smooth function."""

    def __init__(self, smooth_expr: Expression, variable_values: TensorDict):
        """Initialize HVP linear operator.

        Args:
            smooth_expr: The smooth expression to compute Hessian of
            variable_values: Point at which to evaluate the Hessian
        """
        super().__init__()
        self._smooth_expr = smooth_expr
        self._variable_values = variable_values

        n = variable_values.flat_dim()
        self._shape = (n, n)
        self._device = variable_values.to_flat_tensor().device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian @ v using forward-over-reverse autodiff."""

        def grad_dot_v(var_vals: TensorDict) -> torch.Tensor:
            # Compute gradient of smooth_expr
            grad = torch.func.grad(lambda x: self._smooth_expr.evaluate(x))(var_vals)
            return torch.dot(grad.to_flat_tensor(), v)

        # Differentiate grad_dot_v to get Hessian @ v
        hvp_td = torch.func.grad(grad_dot_v)(self._variable_values)
        return hvp_td.to_flat_tensor()
