"""Helper linear operators for splitting representations."""

from functools import partial
from typing import Callable

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
        self.device = self._bias.device

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
        self.device = smooth_expr_vars.to_flat_tensor().device

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
        self._adjoint = self  # Hessian is symmetric
        self.device = variable_values.to_flat_tensor().device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian @ v using forward-over-reverse autodiff."""
        return _pearlmutter_hvp(
            lambda x: self._smooth_expr.evaluate(x), self._variable_values, v
        )


class _SubsampHVPLinOp(LinearOperator):
    """Subsampled Hessian linear operator class.

    Implements a linear operator interface for computing Hessian-vector products
    using a subsampled batch of data. Uses forward-over-reverse automatic
    differentiation for efficient computation.

    Args:
        loss (Callable): Loss function from which subsampled Hessian is constructed.
            Should accept a TensorDict and return a scalar tensor.
        variable_values (TensorDict): Variables at which the subsampled Hessian
            is evaluated.
        X_batch (torch.Tensor): Subsampled data matrix at which the loss is evaluated.
        y_batch (torch.Tensor): Labels tensor at which the loss is evaluated.
        device (torch.device): Device the model parameters live on.

    Attributes:
        device (torch.device): Device on which computations are performed.
    """

    def __init__(
        self,
        loss: Callable[[TensorDict, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        variable_values: TensorDict,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        device: torch.device,
    ):
        """Initialize Subsampled Hessian linear operator.

        Args:
            loss (Callable): Loss function from which subsampled Hessian is constructed.
                Should have signature loss(variable_values, X, y) -> scalar tensor.
            variable_values (TensorDict): Variables at which the subsampled Hessian
                is evaluated.
            X_batch (torch.Tensor): Subsampled data matrix at which the loss
                is evaluated.
            y_batch (torch.Tensor): Labels tensor at which the loss is evaluated.
            device (torch.device): Device the model parameters live on.
        """
        super().__init__()
        self._loss = partial(loss, X=X_batch, y=y_batch)
        self._variable_values = variable_values

        n = variable_values.flat_dim()
        self._shape = (n, n)
        self.device = device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian-vector product using forward-over-reverse autodiff.

        Computes the product of the subsampled Hessian matrix with a vector v
        without explicitly forming the Hessian matrix. Uses automatic differentiation
        to efficiently compute the Hessian-vector product.

        Args:
            v (torch.Tensor): Vector to multiply with the Hessian. Should have
                shape (n,) where n is the flattened dimension of variable_values.

        Returns:
            torch.Tensor: The result of Hessian @ v as a flattened tensor.
        """
        return _pearlmutter_hvp(lambda x: self._loss(x), self._variable_values, v)


def make_subsamp_hvp_linop(
    loss: Callable[[TensorDict, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    variable_values: TensorDict,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    device: torch.device,
) -> LinearOperator:
    """Create a subsampled Hessian-vector product linear operator."""
    return _SubsampHVPLinOp(loss, variable_values, X_batch, y_batch, device)


def _pearlmutter_hvp(
    f: Callable[[TensorDict], torch.Tensor],
    variable_values: TensorDict,
    v: torch.Tensor,
) -> torch.Tensor:
    """Computes Hessian vector product via Pearmutter's trick."""

    def grad_dot_v(var_vals: TensorDict) -> torch.Tensor:
        # Compute gradient of smooth_expr
        grad = torch.func.grad(f)(var_vals)
        return torch.dot(grad.to_flat_tensor(), v)

    # Differentiate grad_dot_v to get Hessian @ v
    hvp_td = torch.func.grad(grad_dot_v)(variable_values)
    return hvp_td.to_flat_tensor()
