"""ADMMSplit class for representing composite objective functions."""

import torch
from linops import LinearOperator, hstack, vstack
from tensordict import merge_tensordicts

from rlaopt.atoms import Atom, AtomDecomposition
from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.linops import _AffineExprLinOp, _HVPLinOp
from rlaopt.splitting.operator_split import _OperatorSplit


class ADMMSplit(_OperatorSplit):
    """Represents a composite objective function suitable for ADMM methods.

    This class is designed for use in ADMM (Alternating Direction Method of Multipliers)
    operator splitting methods.

    Args:
        expr (Expression): An expression object representing the composite function.

    Raises:
        ValueError: If the expression cannot be split into smooth and proxable parts.
    """

    def __init__(self, expr: Expression):
        """Initialize ADMMSplit by validating and splitting the expression.

        Args:
            expr: Expression to split into smooth and proximal components

        Raises:
            ValueError: If expression cannot be split for ADMM
        """
        # Cast to AddExpression for easier splitting
        if not isinstance(expr, AddExpression):
            expr = AddExpression(expr)

        f, decomposed_atoms = self._attempt_split(expr)
        r = [decomposition.atom for decomposition in decomposed_atoms]

        super().__init__(f, r)

        # Store ADMM-specific information
        self._affine_exprs = [
            decomposition.affine_expr for decomposition in decomposed_atoms
        ]
        # We negate the values of b since the decomposition is of the form
        # g(Ax - b) -> g(z) with the constraint Ax - b = z
        # NOTE: We form A_T explicitly to avoid potential issues with
        # automatic adjoint creation in certain linop implementations.
        self._A, self._A_T, self._b, self._b_shapes = _extract_lin_op_and_bias(
            self._affine_exprs, self.f_and_affine_variable_values
        )

    def _attempt_split(
        self, expr: AddExpression
    ) -> tuple[Expression, list[AtomDecomposition]]:
        """Validate and split expression for ADMM.

        Args:
            expr: The expression to split

        Returns:
            tuple: (smooth_expr, decomposed_atoms)

        Raises:
            ValueError: If expression cannot be split for ADMM
        """
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

        # ADMM requires at least one non-smooth term
        if not decomposed_atoms:
            raise ValueError(
                "ADMM splitting requires at least one non-smooth term. "
                "For purely smooth problems, use ProxGrad instead."
            )

        return smooth_part, decomposed_atoms

    @property
    def A(self) -> LinearOperator:
        """Returns the linear operator A used in the decomposition."""
        return self._A

    @property
    def A_T(self) -> LinearOperator:
        """Returns the transpose of the linear operator A used in the decomposition."""
        return self._A_T

    @property
    def b(self) -> torch.Tensor:
        """Returns the bias vector b used in the decomposition."""
        return self._b

    @property
    def variable_values(self) -> TensorDict:
        """Returns the variable values associated with the composite function."""
        td_f_and_affine = self.f_and_affine_variable_values
        td_r = self.r_variable_values
        return merge_tensordicts(td_f_and_affine, td_r)

    @property
    def f_and_affine_variable_values(self) -> TensorDict:
        """Returns the variable values associated with f and the affine constraints.

        This is important for constructing the linear operator A, because
        the affine constraints may depend on variables not present in f, and vice versa.
        """
        td_f = self._f.variable_values
        tds_affine = [expr.variable_values for expr in self._affine_exprs]
        return merge_tensordicts(td_f, *tds_affine)

    @property
    def r_variable_values(self) -> TensorDict:
        """Returns the variable values associated with the proximal function r."""
        tds_r = [r.variable_values for r in self._r]
        return merge_tensordicts(TensorDict({}), *tds_r)

    def init_dual_variables(self) -> TensorDict:
        """Initialize dual variables (u) to zero.

        Returns:
            TensorDict: A dictionary of dual variables initialized to zero.
        """
        dual_vars = {}
        for idx, shape in enumerate(self._b_shapes):
            dual_vars[f"u_{idx}"] = torch.zeros(shape[0], device=self._b.device)
        return TensorDict(dual_vars)

    def hvp_f_ATA_linop(
        self, variable_values: TensorDict, rho: float
    ) -> LinearOperator:
        """Form the Hessian + rho * A^T A linear operator.

        This is useful for inexactly solving the x-subproblem in ADMM methods.

        Args:
            variable_values (TensorDict): A dictionary of variables.
            rho (float): Scaling factor for A^T A term.

        Returns:
            LinearOperator: The combined linear operator.
        """
        hvp_op = _HVPLinOp(self._f, variable_values)
        AT_A = self._A_T @ self._A
        # HACK: add info that might not be preserved by vstack/hstack/summation
        AT_A.device = hvp_op.device
        total_op = hvp_op + rho * AT_A
        total_op.device = hvp_op.device
        total_op.supports_operator_matrix = True
        return total_op


def _extract_lin_op_and_bias(
    affine_exprs: list[Expression],
    f_and_affine_variable_values: TensorDict,
) -> tuple[LinearOperator, LinearOperator, torch.Tensor, list[torch.Size]]:
    """Extracts the linear operator and bias from an affine expression.

    Args:
        affine_exprs (list[Expression]): List of affine expressions of the form Ax - b.
        f_and_affine_variable_values (TensorDict): The variable values associated with
            the smooth part of the objective function along with the affine constraints.

    Returns:
        tuple[LinearOperator, LinearOperator, torch.Tensor, list[torch.Size]]: A tuple
            containing the linear operator A, its transpose A.T, the bias vector b,
            and the shapes of the biases.
    """
    As = []
    ATs = []
    bs = []
    for affine_expr in affine_exprs:
        # Compute the bias of the affine expression
        zero_td = affine_expr.variable_values.apply(torch.zeros_like)
        bias = affine_expr.evaluate(zero_td)

        # Get the linear operator corresponding to the affine expression
        lin_op = _AffineExprLinOp(affine_expr, bias, f_and_affine_variable_values)
        As.append(lin_op)

        # Create the adjoint explicitly
        ATs.append(lin_op.create_adjoint())
        bs.append(bias)

    A = vstack(As)
    A_T = hstack(ATs)
    b = -torch.cat(bs, dim=0)  # Negate to match Ax - b form
    return A, A_T, b, [bias.shape for bias in bs]
