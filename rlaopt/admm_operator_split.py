"""ADMMSplit class for representing composite objective functions."""

import torch
from linops import LinearOperator, vstack
from tensordict import merge_tensordicts

from rlaopt.atoms import Atom, AtomDecomposition
from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict


class _AffineExprLinOp(LinearOperator):
    def __init__(
        self, affine_expr: Expression, bias: torch.Tensor, smooth_expr_vars: TensorDict
    ):
        super().__init__()
        self._affine_expr = affine_expr
        self._bias = bias
        self._unflatten = lambda v: smooth_expr_vars.from_flat_tensor(v)

        input_dim = smooth_expr_vars.flat_dim()
        self._shape = (self._bias.shape[0], input_dim)
        self._device = self._bias.device

    def _matmul_impl(self, v: torch.Tensor):
        v_td = self._unflatten(v)
        return self._affine_expr.evaluate(v_td) - self._bias


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


class ADMMSplit:
    """Represents a composite objective function suitable for ADMM methods.

    This class is designed for use in ADMM (Alternating Direction Method of Multipliers)
    operator splitting methods. It provides access to the smooth part (`f`),
    the non-smooth/proximal part (`r`), their evaluations, the gradient of `f`,
    and the proximal operator of `r`.

    Args:
        expr (Expression): An expression object representing the composite function.

    Raises:
        ValueError: If the expression cannot be split into smooth and proxable parts.
    """

    def __init__(self, expr: Expression):
        """Initializes the ADMMSplit object."""
        # Start by casting to AddExpression for easier splitting
        if not isinstance(expr, AddExpression):
            expr = AddExpression(expr)

        self._f, decomposed_atoms = _attempt_split(expr)
        self._r = [decomposition.atom for decomposition in decomposed_atoms]
        affine_exprs = [decomposition.affine_expr for decomposition in decomposed_atoms]
        # We negate the values of b since the decomposition is of the form
        # g(Ax - b) -> g(z) with the constraint Ax - b = z
        As_and_bs = [
            _extract_lin_op_and_bias(affine_expr, self._f.variable_values)
            for affine_expr in affine_exprs
        ]
        self._As = [A for A, _ in As_and_bs]
        self._bs = [b for _, b in As_and_bs]
        self._A = vstack(self._As)
        self._b = torch.cat(self._bs, dim=0)

    @property
    def f(self) -> Expression:
        """Returns the smooth component of the composite function."""
        return self._f

    @property
    def r(self) -> list[Atom]:
        """Returns the proximal component of the composite function."""
        return self._r

    @property
    def A(self) -> LinearOperator:
        """Returns the linear operator A used in the decomposition."""
        return self._A

    @property
    def b(self) -> torch.Tensor:
        """Returns the bias vector b used in the decomposition."""
        return self._b

    @property
    def variable_values(self) -> TensorDict:
        """Returns the variable values associated with the composite function."""
        td_f = self._f.variable_values
        tds_r = [r.variable_values for r in self._r]
        if not tds_r:
            return td_f
        return merge_tensordicts(td_f, *tds_r)

    @property
    def f_variable_values(self) -> TensorDict:
        """Returns the variable values associated with the smooth function f."""
        return self._f.variable_values

    @property
    def r_variable_values(self) -> TensorDict:
        """Returns the variable values associated with the proximal function r."""
        tds_r = [r.variable_values for r in self._r]
        if not tds_r:
            # Return empty TensorDict if no r atoms
            return TensorDict()
        if len(tds_r) == 1:
            return tds_r[0]
        return merge_tensordicts(*tds_r)

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
        hvp_op = _HVPLinOp(self._f, variable_values)
        return hvp_op @ v

    def hvp_f_ATA_linop(
        self, variable_values: TensorDict, rho: float, sigma: float
    ) -> LinearOperator:
        """Form the Hessian + rho * A^T A + sigma * I linear operator.

        Args:
            variable_values (TensorDict): A dictionary of variables.
            rho (float): Scaling factor for A^T A term.
            sigma (float): Scaling factor for the identity term.

        Returns:
            LinearOperator: The combined linear operator.
        """
        hvp_op = _HVPLinOp(self._f, variable_values)
        AT_A = self._A.T @ self._A
        return hvp_op + rho * AT_A + sigma

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


def _attempt_split(expr: AddExpression) -> tuple[Expression, list[AtomDecomposition]]:
    smooth_part = expr.get_smooth_part()
    non_smooth_exprs = expr.get_non_smooth_exprs()

    # Verify that all variables are 1d
    var_shapes = smooth_part.get_variable_shapes()
    for non_smooth_expr in non_smooth_exprs:
        var_shapes.update(non_smooth_expr.get_variable_shapes())

    if any(len(shape) != 1 for shape in var_shapes.values()):
        raise ValueError(
            "All variables must be 1-dimensional for ADMM operator splitting."
        )

    # All non-smooth terms must be decomposable
    decomposed_atoms = []
    for non_smooth_expr in non_smooth_exprs:
        if not isinstance(non_smooth_expr, Atom):
            raise ValueError(
                "All non-smooth terms must be instances of Atom for ADMM splitting."
            )
        decomposition = non_smooth_expr.decompose()
        if decomposition is None:
            raise ValueError(
                "All non-smooth terms must be decomposable for ADMM splitting."
            )
        decomposed_atoms.extend(decomposition)
    return smooth_part, decomposed_atoms


def _extract_lin_op_and_bias(
    affine_expr: Expression,
    smooth_expr_vars: TensorDict,
) -> tuple[_AffineExprLinOp, torch.Tensor]:
    """Extracts the linear operator and bias from an affine expression.

    Args:
        affine_expr (Expression): An affine expression of the form Ax - b.
        smooth_expr_vars (TensorDict): The variable values associated with the
            smooth part of the objective function.

    Returns:
        tuple[LinearOperator, torch.Tensor]: A tuple containing the linear operator A
        and the bias vector b.
    """
    # Evaluate at zero to get the bias
    zero_td = affine_expr.variable_values.clone()
    for key in zero_td.keys():
        zero_td[key] = torch.zeros_like(zero_td[key])
    bias = affine_expr.evaluate(zero_td)

    lin_op = _AffineExprLinOp(affine_expr, bias, smooth_expr_vars)
    # Negate bias to match the standard form Ax - b
    return lin_op, -bias
