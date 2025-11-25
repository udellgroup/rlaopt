"""Helper linear operators for splitting representations."""

import torch
from linops import LinearOperator

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict


def _apply_batched(fn, v: torch.Tensor) -> torch.Tensor:
    """Apply function to vector or batch of vectors (matrix columns).

    Args:
        fn: Function to apply to each column
        v: Input tensor (1D vector or 2D matrix)

    Returns:
        Result tensor with same dimensionality as input
    """
    squeeze_output = v.ndim == 1
    if squeeze_output:
        v = v.unsqueeze(-1)

    result = torch.func.vmap(fn, in_dims=1, out_dims=1)(v)
    return result.squeeze(-1) if squeeze_output else result


class _AffineExprLinOp(LinearOperator):
    """Linear operator for affine expressions of the form Ax - b."""

    def __init__(
        self, affine_expr: Expression, bias: torch.Tensor, smooth_expr_vars: TensorDict
    ):
        super().__init__()
        self._affine_expr = affine_expr
        self._smooth_expr_vars = smooth_expr_vars
        self._zero_vars = smooth_expr_vars.apply(torch.zeros_like)

        input_dim = smooth_expr_vars.flat_dim()
        self._shape = (bias.shape[0], input_dim)
        self.device = bias.device
        self.supports_operator_matrix = True

    def _matmul_impl(self, v: torch.Tensor):
        """Compute A @ v using JVP (Jacobian-vector product)."""

        def jvp_column(v_col: torch.Tensor) -> torch.Tensor:
            v_td = self._smooth_expr_vars.from_flat_tensor(v_col)
            _, tangent = torch.func.jvp(
                lambda x: self._affine_expr.evaluate(x),
                (self._zero_vars,),
                (v_td,),
            )
            return tangent

        return _apply_batched(jvp_column, v)

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
        self._zero_vars = smooth_expr_vars.apply(torch.zeros_like)

        m, n = forward_shape
        self._shape = (n, m)
        self.device = smooth_expr_vars.to_flat_tensor().device
        self.supports_operator_matrix = True

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute A.T @ v using VJP (vector-Jacobian product)."""

        def vjp_column(v_col: torch.Tensor) -> torch.Tensor:
            _, vjp_fn = torch.func.vjp(
                lambda x: self._affine_expr.evaluate(x), self._zero_vars
            )
            return vjp_fn(v_col)[0].to_flat_tensor()

        return _apply_batched(vjp_column, v)


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
        self._variable_values_flat = variable_values.to_flat_tensor()

        n = variable_values.flat_dim()
        self._shape = (n, n)
        self._adjoint = self  # Hessian is symmetric
        self.device = variable_values.to_flat_tensor().device
        self.supports_operator_matrix = True

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian @ v using forward-over-reverse autodiff.

        Works directly with flat tensors to avoid TensorDict complications.
        """

        def hvp_column(v_col: torch.Tensor) -> torch.Tensor:
            # Define function in terms of flat tensor
            def f_flat(x_flat: torch.Tensor) -> torch.Tensor:
                x_td = self._variable_values.from_flat_tensor(x_flat)
                return self._smooth_expr.evaluate(x_td)

            # Compute gradient with respect to flat tensor
            def grad_f_flat(x_flat: torch.Tensor) -> torch.Tensor:
                grad_td = torch.func.grad(lambda x: f_flat(x))(x_flat)
                return grad_td

            # Compute JVP: d/dx[grad(f(x))] @ v
            _, hvp = torch.func.jvp(
                grad_f_flat, (self._variable_values_flat,), (v_col,)
            )
            return hvp

        return _apply_batched(hvp_column, v)
