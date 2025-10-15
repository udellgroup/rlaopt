"""Expression module for symbolic optimization modeling.

This module provides the core expression classes for building symbolic mathematical
expressions that can be evaluated, differentiated, and optimized. Expressions are
built compositionally through operator overloading and can represent optimization
objectives, constraints, and models.

Classes:
    Expression: Abstract base class for all expressions.
    Variable: Leaf optimization variable wrapping a torch.nn.Parameter.
    ConstExpression: Constant value expression.
    AddExpression: Sum of multiple expressions.
    ProductExpression: Product of multiple expressions.
    UnaryOpExpression: Unary operation applied to an expression.
"""

import inspect
from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, Union

import cvxpy as cp
import torch

from rlaopt._typing import TensorDict
from rlaopt.settings import VAR_PREFIX
from rlaopt.utils import tensor_dict_ops as dict_ops
from rlaopt.utils.counter import get_id


# ===============================
# Helper Functions
# ===============================
def _to_expr(val) -> "Expression":
    """Convert a value to an Expression if it isn't already.

    Args:
        val: Value to convert. Can be an Expression, float, int, or torch.Tensor.

    Returns:
        Expression: The converted expression.

    Raises:
        TypeError: If val cannot be converted to an Expression.

    """
    if isinstance(val, Expression):
        return val
    if isinstance(val, (float, int, torch.Tensor)):
        return ConstExpression(val)
    raise TypeError(f"Cannot convert {type(val)} to Expression")


# ===============================
# Base Expression
# ===============================
class Expression(torch.nn.Module, ABC):
    """Abstract base class for all expressions.

    Expression extends torch.nn.Module to provide automatic parameter tracking,
    gradient computation, and device management. All concrete expression types
    (Variable, AddExpression, etc.) inherit from this base class.

    Expressions support operator overloading for natural mathematical syntax:
        - Arithmetic: +, -, *, /, @, **
        - Comparisons: Used in constraints (future)
        - Composition: Expressions can be nested arbitrarily

    Attributes:
        None (abstract class)
    """

    def __init__(self):
        """Initialize the Expression base class."""
        super().__init__()

    @abstractmethod
    def is_smooth(self) -> bool:
        """Check if the expression is smooth (differentiable everywhere).

        Smoothness is important for choosing optimization algorithms. Smooth
        expressions can use gradient-based methods, while non-smooth expressions
        require specialized algorithms like proximal methods or subgradient methods.

        Returns:
            bool: True if the expression is smooth, False otherwise.

        Examples:
            >>> x = Variable((5,))
            >>> x.is_smooth()
            True
            >>> from rlaopt.atoms import L1Norm
            >>> l1 = L1Norm(x)
            >>> l1.is_smooth()
            False
        """
        pass

    @abstractmethod
    def is_proxable(self) -> bool:
        """Check if the expression has a computable proximal operator.

        The proximal operator is used in proximal gradient methods and ADMM
        for non-smooth optimization. An expression is proxable if its proximal
        operator can be computed efficiently in closed form.

        Returns:
            bool: True if the expression is proxable, False otherwise.

        Examples:
            >>> from rlaopt.atoms import L1Norm
            >>> x = Variable((5,))
            >>> l1 = L1Norm(x)
            >>> l1.is_proxable()
            True
        """
        pass

    @abstractmethod
    def to_cvxpy(self) -> cp.Expression:
        """Convert the expression to a CVXPY expression.

        This allows verification of custom solvers against CVXPY's trusted
        implementations and enables fallback to CVXPY for difficult problems.

        Returns:
            cp.Expression: A CVXPY expression representing this expression.

        Note:
            This method may be deprecated in future versions as the library
            diverges from CVXPY's architecture.
        """
        pass

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """Evaluate the expression using current parameter values.

        This method evaluates the expression with the current values of all
        registered parameters. Additional keyword arguments can be passed
        for expressions that require extra context (e.g., data for loss functions).

        Args:
            **kwargs: Additional keyword arguments for evaluating the expression.

        Returns:
            torch.Tensor: The evaluated result.

        Examples:
            >>> x = Variable((5,))
            >>> x.value.data = torch.ones(5)
            >>> result = x.forward()
            >>> torch.equal(result, torch.ones(5))
            True
        """
        pass

    def evaluate(self, params: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Evaluate the expression at specified parameter values.

        Unlike forward(), this method evaluates the expression at parameter
        values different from those currently stored, without modifying the
        stored parameters. Useful for line searches, parameter exploration, etc.

        Args:
            params: Dictionary mapping parameter names to their values.
            **kwargs: Additional keyword arguments for evaluation.

        Returns:
            torch.Tensor: The evaluated result.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> x.value.data = torch.zeros(5)
            >>> result = x.evaluate({'exprs.0.x': torch.ones(5)})
            >>> torch.equal(result, torch.ones(5))
            True
        """
        return torch.func.functional_call(self, params, args=None, kwargs=kwargs)

    def params_dict(self) -> TensorDict:
        """Get all parameters as a dictionary.

        Returns:
            TensorDict: Dictionary of parameter names to parameter tensors.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> params = x.params_dict()
            >>> 'x' in str(params.keys())
            True
        """
        return dict(self.named_parameters())

    def update_params(self, params_dict: TensorDict):
        """Update parameters from a dictionary.

        Args:
            params_dict: Dictionary mapping parameter names to new values.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> x.update_params({'x': torch.ones(5)})
            >>> torch.equal(x.value, torch.ones(5))
            True
        """
        self.load_state_dict(params_dict, strict=False)

    def expr_convert_params(
        self, params_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert parameter names from another expression to match this one.

        Given parameters from another Expression whose leaf tensor shapes match
        but have different names, returns a dictionary with names consistent
        with the current expression. Useful for transferring parameters between
        similar expressions.

        Args:
            params_dict: Parameters from another expression.

        Returns:
            dict[str, torch.Tensor]: Parameters with names matching this expression.
        """
        return dict_ops.relabel_from_template(params_dict, self.params)

    @property
    def params(self) -> TensorDict:
        """Get parameters as a dictionary.

        Returns:
            TensorDict: Dictionary of parameter names to parameter tensors.
        """
        return self.params_dict()

    # ----------------------
    # Centralized operator overloads
    # ----------------------
    def __add__(self, other):
        """Add two expressions or an expression and a scalar.

        Args:
            other: Expression, float, or int to add.

        Returns:
            AddExpression: Sum of self and other.
        """
        return AddExpression(self, other)

    def __radd__(self, other):
        """Add a scalar and an expression (reverse operation).

        Args:
            other: Float or int to add.

        Returns:
            AddExpression: Sum of other and self.
        """
        return AddExpression(other, self)

    def __sub__(self, other):
        """Subtract an expression or scalar from this expression.

        Args:
            other: Expression, float, or int to subtract.

        Returns:
            AddExpression: Difference of self and other.
        """
        return AddExpression(self, -other)

    def __rsub__(self, other):
        """Subtract this expression from a scalar (reverse operation).

        Args:
            other: Float or int to subtract from.

        Returns:
            AddExpression: Difference of other and self.
        """
        return AddExpression(other, -self)

    def __neg__(self):
        """Negate this expression.

        Returns:
            ProductExpression: Negation of self.
        """
        return ProductExpression(ConstExpression(-1.0), self)

    def __mul__(self, other):
        """Multiply this expression by another (elementwise).

        Args:
            other: Expression, float, or int to multiply.

        Returns:
            ProductExpression: Elementwise product of self and other.
        """
        return ProductExpression(self, other, matmul=False)

    def __rmul__(self, other):
        """Multiply a scalar by this expression (reverse operation).

        Args:
            other: Float or int to multiply.

        Returns:
            ProductExpression: Elementwise product of other and self.
        """
        return ProductExpression(other, self, matmul=False)

    def __truediv__(self, other):
        """Divide this expression by a scalar.

        Args:
            other: Float or int to divide by.

        Returns:
            ProductExpression: Result of division.
            NotImplemented: If other is not a scalar.
        """
        if isinstance(other, (int, float)):
            return ProductExpression(self, ConstExpression(1.0 / other), matmul=False)
        return NotImplemented

    def __matmul__(self, other):
        """Matrix multiply this expression by another.

        Args:
            other: Expression, float, or int to multiply.

        Returns:
            ProductExpression: Matrix product of self and other.
        """
        return ProductExpression(self, other, matmul=True)

    def __rmatmul__(self, other):
        """Matrix multiply a value by this expression (reverse operation).

        Args:
            other: Expression, float, or int to multiply.

        Returns:
            ProductExpression: Matrix product of other and self.
        """
        return ProductExpression(other, self, matmul=True)

    def __pow__(self, exponent):
        """Raise this expression to a power (elementwise).

        Args:
            exponent: Power to raise to.

        Returns:
            UnaryOpExpression: Result of exponentiation.
        """
        return UnaryOpExpression(self, lambda t: torch.pow(t, exponent))


class ConstExpression(Expression):
    """Constant value expression.

    Represents a constant (non-trainable) value in an expression tree.
    Constants are stored as buffers rather than parameters, so they
    don't receive gradients and don't appear in parameter optimization.

    Args:
        value: The constant value (float, int, or torch.Tensor).

    Attributes:
        _value: The constant value stored as a buffer.

    Examples:
        >>> c = ConstExpression(3.14)
        >>> c.forward()
        tensor(3.1400)
        >>> c2 = ConstExpression(torch.ones(5))
        >>> c2.forward().shape
        torch.Size([5])
    """

    def __init__(self, value: Union[float, int, torch.Tensor]):
        """Initialize a constant expression.

        Args:
            value: The constant value to store.
        """
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        # register as buffer so it is visible but not a parameter
        self.register_buffer("_value", value)

    @property
    def value(self):
        """Get the constant value.

        Returns:
            torch.Tensor: The constant value.
        """
        return getattr(self, "_value")

    def is_smooth(self) -> bool:
        """Constants are smooth (trivially differentiable).

        Returns:
            bool: Always True.
        """
        return True

    def is_proxable(self) -> bool:
        """Constants are proxable (proximal operator is identity).

        Returns:
            bool: Always True.
        """
        return True

    def forward(self) -> torch.Tensor:
        """Evaluate the constant (returns itself).

        Returns:
            torch.Tensor: The constant value.
        """
        return self.value

    def to_cvxpy(self) -> cp.Expression:
        """Convert to CVXPY constant.

        Returns:
            cp.Constant: CVXPY constant expression.
        """
        return cp.Constant(self.value.detach().cpu().numpy())

    def __neg__(self):
        """Negate the constant (keeps it as a constant).

        Returns:
            ConstExpression: Negated constant.
        """
        return ConstExpression(-self.value)


class NAryOperatorExpression(Expression, ABC):
    """Base class for n-ary operations (operations on multiple expressions).

    Provides infrastructure for expressions that operate on multiple sub-expressions,
    such as addition and multiplication. Handles parameter tracking, evaluation,
    and keyword argument filtering automatically.

    Args:
        *exprs: Variable number of expressions to operate on.

    Attributes:
        exprs: ModuleList of sub-expressions.
        _expr_signatures: Cached function signatures for kwargs filtering.
    """

    @abstractmethod
    def op(self, exprs: list[torch.Tensor], lib):
        """Apply the operator to evaluated tensors.

        Args:
            exprs: List of evaluated tensor expressions.
            lib: Library to use (torch or cp for CVXPY).

        Returns:
            torch.Tensor or cp.Expression: Result of the operation.
        """
        pass

    def __init__(self, *exprs):
        """Initialize with variable number of expressions.

        Args:
            *exprs: Expressions to combine. Last argument can be None.

        Raises:
            ValueError: If no expressions provided.
        """
        super().__init__()
        if len(exprs) == 0:
            raise ValueError(f"{self.__class__.__name__} requires at least one operand")
        else:
            if exprs[-1] is not None:
                self.exprs = torch.nn.ModuleList([_to_expr(e) for e in exprs])
            else:
                # If last expr is None, ignore it
                # This is needed for operator_split in AddExpression
                self.exprs = torch.nn.ModuleList([_to_expr(e) for e in exprs[:-1]])

        # Cache expression signatures for kwargs filtering
        self._expr_signatures = {e: inspect.signature(e.forward) for e in self.exprs}

    def is_smooth(self) -> bool:
        """Check if all sub-expressions are smooth.

        Returns:
            bool: True if all sub-expressions are smooth.
        """
        return all(expr.is_smooth() for expr in self.exprs)

    def forward(self, **kwargs) -> torch.Tensor:
        """Evaluate the operation with current parameters.

        Evaluates each sub-expression and applies the operator. Automatically
        filters kwargs to only pass relevant arguments to each sub-expression.

        Args:
            **kwargs: Keyword arguments for sub-expression evaluation.

        Returns:
            torch.Tensor: Result of the operation.
        """
        vals = [self._eval_expr(expr, **kwargs) for expr in self.exprs]
        return self.op(vals, torch)

    def _eval_expr(self, expr: Expression, **kwargs):
        """Evaluate a sub-expression with filtered kwargs.

        Filters kwargs to only those accepted by the expression's forward method.

        Args:
            expr: Expression to evaluate.
            **kwargs: All available keyword arguments.

        Returns:
            torch.Tensor: Evaluated expression.
        """
        sig = self._expr_signatures[expr]
        relevant_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return expr.forward(**relevant_kwargs)

    def to_cvxpy(self):
        """Convert to CVXPY expression.

        Returns:
            cp.Expression: CVXPY representation of the operation.
        """
        return self.op([expr.to_cvxpy() for expr in self.exprs], cp)


# ===============================
# AddExpression (sum of exprs)
# ===============================


class AddExpression(NAryOperatorExpression):
    """Sum of multiple expressions.

    Represents the sum of two or more expressions. Uses a hybrid approach
    for efficiency: fast vectorized path when all shapes match, correct
    broadcasting path when shapes differ (e.g., scalar + vector).

    Args:
        left_or_exprs: First expression or list of expressions.
        right: Second expression (optional, can be None).

    Attributes:
        _num_non_smooth_exprs: Count of non-smooth terms (for proxability).
        _prox: Proximal operator function (if proxable).

    Examples:
        >>> x = Variable((5,))
        >>> y = Variable((5,))
        >>> z = x + y + 10
        >>> isinstance(z, AddExpression)
        True
    """

    def __init__(self, left_or_exprs, right=None):
        """Initialize sum expression.

        Args:
            left_or_exprs: First expression.
            right: Second expression (can be None for operator splitting).
        """
        super().__init__(left_or_exprs, right)
        self._num_non_smooth_exprs = self._count_non_smooth_terms()
        self._prox = self._build_prox()

    def op(self, exprs, lib):
        """Apply the addition operator with hybrid optimization.

        Uses fast vectorized stack+sum when all shapes match, or sequential
        addition with broadcasting when shapes differ.

        Args:
            exprs: List of evaluated tensor expressions.
            lib: Library to use (torch or cvxpy).

        Returns:
            torch.Tensor or cp.Expression: Sum of all expressions.

        Raises:
            ValueError: If expression list is empty.
        """
        if not exprs:
            raise ValueError("AddExpression requires at least one operand")
        if len(exprs) == 1:
            return exprs[0]

        if lib is torch:
            # Check if all tensors have the same shape
            first_shape = exprs[0].shape
            all_same_shape = all(e.shape == first_shape for e in exprs[1:])

            if all_same_shape:
                # Fast path: all same shape - use vectorized stack + sum
                stacked = lib.stack(exprs, dim=0)
                return lib.sum(stacked, dim=0)
            else:
                # Correct path: mixed shapes - use broadcasting
                result = exprs[0]
                for e in exprs[1:]:
                    result = result + e
                return result
        else:
            # CVXPY: use reduce with addition
            return reduce(lambda a, b: a + b, exprs)

    def is_proxable(self):
        """Check if sum is proxable.

        A sum is proxable if:
        1. All non-smooth terms are proxable, AND
        2. Non-smooth terms operate on disjoint parameter sets.

        Returns:
            bool: True if the sum is proxable.
        """
        non_smooth_exprs = [e for e in self.exprs if not e.is_smooth()]

        # All non-smooth terms must be proxable
        if any(not expr.is_proxable() for expr in non_smooth_exprs):
            return False

        # Check for parameter overlap
        seen_params = set()
        for expr in non_smooth_exprs:
            expr_params = set(expr.parameters())
            if seen_params & expr_params:  # intersection
                return False
            seen_params.update(expr_params)

        return True

    def operator_split(self):
        """Split sum into smooth and non-smooth parts.

        Separates the sum into two expressions: one containing all smooth terms
        and one containing all non-smooth terms. Useful for proximal algorithms.

        Returns:
            tuple: (smooth_expr, non_smooth_expr) where either can be None.

        Examples:
            >>> from rlaopt.atoms import L1Norm, SumSquares
            >>> x = Variable((5,))
            >>> expr = SumSquares(x) + L1Norm(x)
            >>> smooth, non_smooth = expr.operator_split()
            >>> smooth.is_smooth()
            True
            >>> non_smooth.is_smooth()
            False
        """
        smooth = [e for e in self.exprs if e.is_smooth()]
        non_smooth = [e for e in self.exprs if not e.is_smooth()]
        smooth_expr = AddExpression(*smooth) if smooth else None
        non_smooth_expr = AddExpression(*non_smooth) if non_smooth else None

        return smooth_expr, non_smooth_expr

    def prox(self, location, prox_scaling):
        """Compute proximal operator of the sum.

        Args:
            location: Point at which to evaluate proximal operator.
            prox_scaling: Scaling factor for proximal operator.

        Returns:
            torch.Tensor or dict: Result of proximal operator.

        Raises:
            NotImplementedError: If the sum is not proxable.
        """
        return self._prox(location, prox_scaling)

    @property
    def num_non_smooth_exprs(self):
        """Get count of non-smooth terms.

        Returns:
            int: Number of non-smooth expressions in the sum.
        """
        return self._num_non_smooth_exprs

    def _build_prox(self):
        """Build the proximal operator function.

        Returns:
            Callable: Proximal operator function.
        """
        if not self.is_proxable():

            def prox(location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
                raise NotImplementedError("Expression is not proxable")

            return prox

        elif self._num_non_smooth_exprs == 1:
            proxes = self._get_proxes()
            return proxes[0]

        else:
            proxes = self._get_proxes()

            def prox(
                location: dict[str, torch.Tensor], prox_scaling: float
            ) -> dict[str, torch.Tensor]:
                return {
                    name: prox_fn(loc, prox_scaling)
                    for (name, loc), prox_fn in zip(location.items(), proxes)
                }

            return prox

    def _count_non_smooth_terms(self):
        """Count non-smooth expressions in the sum.

        Returns:
            int: Number of non-smooth terms.
        """
        return sum(1 for expr in self.exprs if not expr.is_smooth())

    def _get_proxes(self) -> list[Callable[[torch.Tensor, float], torch.Tensor]]:
        """Get proximal operators of non-smooth expressions.

        Returns:
            list: List of proximal operator functions.
        """
        proxes = []
        for expr in self.exprs:
            if not expr.is_smooth():
                proxes.append(expr.prox)
        return proxes

    def to_cvxpy(self) -> cp.Expression:
        """Convert to CVXPY expression.

        Returns:
            cp.Expression: CVXPY sum expression.
        """
        return super().to_cvxpy()


# ===============================
# ProductExpression (* and @)
# ===============================


class ProductExpression(NAryOperatorExpression):
    """Product of multiple expressions.

    Represents either elementwise multiplication (*) or matrix multiplication (@)
    of multiple expressions. Validates that only simple expressions (Variables
    and Constants) are multiplied together to avoid ambiguous differentiation.

    Args:
        *exprs: Expressions to multiply.
        matmul: If True, use matrix multiplication; if False, elementwise.

    Attributes:
        matmul: Whether to use matrix multiplication.

    Examples:
        >>> x = Variable((5,))
        >>> y = Variable((5,))
        >>> z = x * y  # Elementwise
        >>> A = Variable((3, 4))
        >>> b = Variable((4,))
        >>> c = A @ b  # Matrix multiplication
    """

    def __init__(self, *exprs, matmul: bool = False):
        """Initialize product expression.

        Args:
            *exprs: Expressions to multiply.
            matmul: Whether to use matrix multiplication.

        Raises:
            TypeError: If multiplying complex parameterized expressions.
        """
        self.matmul = matmul
        super().__init__(*exprs)
        self._validate()

    def _validate(self):
        """Validate that multiplication is allowed.

        Only allows multiplication of Variables and Constants, or expressions
        built purely from Variables and Constants using basic operations.

        Raises:
            TypeError: If validation fails.
        """
        param_exprs = [e for e in self.exprs if list(e.parameters())]

        if len(param_exprs) > 1:
            if not all(self._is_var_or_const_tree(e) for e in param_exprs):
                raise TypeError(
                    "Cannot multiply two arbitrary parameterized Expressions. "
                    "Only Variables and Constants can be multiplied together."
                )

    def _is_var_or_const_tree(self, expr: Expression) -> bool:
        """Check if expression tree contains only Variables and Constants.

        Args:
            expr: Expression to check.

        Returns:
            bool: True if tree contains only simple expressions.
        """
        if isinstance(expr, (Variable, ConstExpression)):
            return True
        if isinstance(expr, (ProductExpression, AddExpression)):
            return all(self._is_var_or_const_tree(child) for child in expr.exprs)
        if isinstance(expr, UnaryOpExpression):
            return self._is_var_or_const_tree(expr.operand)
        return False

    def op(self, exprs, lib):
        """Apply multiplication operator.

        Args:
            exprs: List of evaluated tensor expressions.
            lib: Library to use (torch or cvxpy).

        Returns:
            torch.Tensor or cp.Expression: Product of all expressions.

        Raises:
            ValueError: If expression list is empty.
        """
        if not exprs:
            raise ValueError("ProductExpression requires at least one operand")
        if len(exprs) == 1:
            return exprs[0]
        if lib is torch:
            res = exprs[0]
            for e in exprs[1:]:
                res = lib.matmul(res, e) if self.matmul else res * e
            return res
        else:
            if self.matmul:
                return reduce(lambda a, b: a @ b, exprs)
            else:
                return reduce(lambda a, b: a * b, exprs)

    def is_smooth(self):
        """Products are always smooth.

        Returns:
            bool: Always True.
        """
        return True

    def is_proxable(self):
        """Products are not proxable.

        Returns:
            bool: Always False.
        """
        return False

    def prox(self, location, prox_scaling):
        """Products don't have proximal operators.

        Args:
            location: Unused.
            prox_scaling: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("ProductExpression is not proxable")

    def to_cvxpy(self):
        """Convert to CVXPY expression.

        Returns:
            cp.Expression: CVXPY product expression.
        """
        return super().to_cvxpy()


# ===============================
# Unary ops
# ===============================
class UnaryOpExpression(Expression):
    """Unary operation applied to an expression.

    Represents the application of a unary function (e.g., abs, exp, sum)
    to another expression. The operation is stored as a lambda function.

    Args:
        operand: Expression to apply operation to.
        op: Unary function to apply (torch.Tensor -> torch.Tensor).

    Attributes:
        operand: The sub-expression.
        _op: The unary operation function.

    Examples:
        >>> x = Variable((5,))
        >>> x_squared = x ** 2  # UnaryOpExpression with torch.pow
        >>> x_sum = x.sum()  # UnaryOpExpression with torch.sum
    """

    def __init__(
        self,
        operand: Expression | Union[float, int, torch.Tensor],
        op: Callable[[torch.Tensor], torch.Tensor],
    ):
        """Initialize unary operation expression.

        Args:
            operand: Expression or value to operate on.
            op: Function to apply.
        """
        super().__init__()
        self.operand = _to_expr(operand)
        self.add_module("_operand", self.operand)
        self._op = op

    def forward(self, **kwargs) -> torch.Tensor:
        """Evaluate the unary operation.

        Args:
            **kwargs: Keyword arguments for operand evaluation.

        Returns:
            torch.Tensor: Result of applying operation to operand.
        """
        val = self.operand.forward(**kwargs)
        return self._op(val)

    def is_smooth(self) -> bool:
        """Smoothness depends on the operand.

        Returns:
            bool: True if operand is smooth.
        """
        return self.operand.is_smooth()

    def is_proxable(self) -> bool:
        """Unary operations are generally not proxable.

        Returns:
            bool: Always False.
        """
        return False

    def to_cvxpy(self):
        """CVXPY conversion not implemented for generic unary ops.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("to_cvxpy not implemented for UnaryOpExpression")

    def sum(self, dim=None):
        """Create a sum operation.

        Args:
            dim: Dimension to sum over (None for all dimensions).

        Returns:
            UnaryOpExpression: Expression computing the sum.
        """
        return UnaryOpExpression(self, lambda t: torch.sum(t, dim=dim))


# ===============================
# Variable (leaf)
# ===============================
class Variable(Expression):
    """Leaf optimization variable wrapping a torch.nn.Parameter.

    Variable represents a trainable parameter in an optimization problem.
    It extends Expression to provide automatic differentiation, device
    management, and integration with PyTorch's optimization ecosystem.

    Variables register their parameters with meaningful names (not just 'value')
    to improve debugging and state_dict readability.

    Args:
        *size_or_tensor: Either a size tuple (e.g., (5,) or (5, 10)) or an
            existing torch.Tensor to wrap.
        requires_grad: Whether the parameter should track gradients.
        var_id: Optional custom ID for the variable.
        name: Optional custom name for the variable.
        dtype: Data type for the variable tensor.
        device: Device to place the variable on.

    Attributes:
        _id: Unique identifier for this variable.
        _name: Name of the variable (used as parameter name).

    Examples:
        >>> # Create from size
        >>> x = Variable((5,), name='weights')
        >>> x.value.shape
        torch.Size([5])

        >>> # Create matrix variable
        >>> A = Variable((3, 4,), name='matrix')
        >>> A.value.shape
        torch.Size([3, 4])

        >>> # Create from existing tensor
        >>> data = torch.randn(10)
        >>> y = Variable(data, name='initialized')

        >>> # State dict uses meaningful names
        >>> expr = x + y
        >>> list(expr.state_dict().keys())
        ['exprs.0.weights', 'exprs.1.initialized']
    """

    def __init__(
        self,
        size_or_tensor: tuple[int, ...] | torch.Tensor,
        requires_grad: bool = True,
        var_id: int | None = None,
        name: str | None = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        """Initialize a new Variable instance.

        Args:
            size_or_tensor (tuple[int,...] | torch.Tensor): Tensor size or tensor.
            requires_grad: Whether to track gradients.
            var_id: Optional custom ID.
            name: Optional custom name.
            dtype: Data type for the tensor.
            device: Device to place tensor on.
        """
        super().__init__()

        # If input is tensor, data is tensor
        if isinstance(size_or_tensor, torch.Tensor):
            data = size_or_tensor
        # If input is tuple of shapes, data is zeros tensor of appropriate shape.
        elif isinstance(size_or_tensor, tuple):
            # Create tensor
            data = torch.zeros(size_or_tensor, dtype=dtype, device=device)
        # Raise type error otherwise.
        else:
            raise TypeError(
                f"size must be tuple[int, ...] or torch.Tensor, "
                f"got {type(size_or_tensor)}"
            )

        self._set_id_and_name(var_id, name)

        # Register parameter with variable's name for better state_dict readability
        self.register_parameter(self._name, torch.nn.Parameter(data, requires_grad))

    def _set_id_and_name(self, var_id=None, name=None):
        """Set ID and name attributes.

        Args:
            var_id: Optional custom ID (generates unique ID if None).
            name: Optional custom name (generates default if None).

        Raises:
            TypeError: If name is not a string.
        """
        # Set ID
        if var_id is None:
            self._id = get_id()
        else:
            self._id = var_id

        # Set name
        if name is None:
            self._name = f"{VAR_PREFIX}{self._id}"
        elif isinstance(name, str):
            self._name = name
        else:
            raise TypeError(f"Expected name to be a string, got {type(name)} instead.")

    @property
    def value(self) -> torch.nn.Parameter:
        """Get the parameter value.

        Returns the underlying torch.nn.Parameter using the variable's name.
        This allows accessing parameters as x.value while storing them with
        meaningful names in the state_dict.

        Returns:
            torch.nn.Parameter: The parameter tensor.

        Examples:
            >>> x = Variable((5,), name='alpha')
            >>> x.value.data = torch.ones(5)
            >>> x.value
            Parameter containing:
            tensor([1., 1., 1., 1., 1.], requires_grad=True)
        """
        return getattr(self, self._name)

    @value.setter
    def value(self, val: torch.Tensor):
        """Set the parameter value.

        Replaces the entire parameter with a new torch.nn.Parameter wrapping
        the provided tensor. Preserves the requires_grad setting from the
        existing parameter if it exists.

        Args:
            val: New tensor value to wrap in a Parameter.

        Examples:
            >>> x = Variable((5,), name='beta')
            >>> x.value = torch.randn(5)
            >>> x.value.shape
            torch.Size([5])
        """
        requires_grad = getattr(self.value, "requires_grad", True)
        self.register_parameter(self._name, torch.nn.Parameter(val, requires_grad))

    @property
    def id(self) -> int:
        """Get the unique identifier of the variable.

        Returns:
            int: The variable's unique ID.
        """
        return self._id

    @property
    def name(self) -> str:
        """Get the name of the variable.

        Returns:
            str: The variable's name.
        """
        return self._name

    def to_cvxpy(self) -> cp.Variable:
        """Convert to CVXPY variable.

        Returns:
            cp.Variable: CVXPY variable with same shape and name.
        """
        return cp.Variable(shape=self.value.shape, name=self.name, var_id=self.id)

    def __repr__(self):
        """Full representation of the Variable.

        Returns:
            str: Detailed string representation including all attributes.

        Examples:
            >>> x = Variable(5, name='weights')
            >>> repr(x)
            "Variable(name='weights', id='...', shape=(5,), dtype=torch.float32, ...)"
        """
        info_components = [
            f"Variable(name='{self.name}'",
            f"id='{self.id}'",
            f"shape={tuple(self.value.shape)}",
            f"dtype={self.value.dtype}",
            f"device='{self.value.device}'",
            f"requires_grad={self.value.requires_grad}",
        ]
        info = ", ".join(info_components)
        return info + ")"

    def __str__(self):
        """Shortened representation of the Variable.

        Returns:
            str: Brief string representation.

        Examples:
            >>> x = Variable((5,), name='weights')
            >>> str(x)
            "Variable 'weights' with shape torch.Size([5])"
        """
        return f"Variable '{self.name}' with shape {self.value.shape}"

    def is_smooth(self):
        """Variables are smooth (identity function is differentiable).

        Returns:
            bool: Always True.
        """
        return True

    def is_proxable(self):
        """Variables don't have meaningful proximal operators.

        Returns:
            bool: Always False.
        """
        return False

    def forward(self) -> torch.Tensor:
        """Evaluate the variable (returns its current value).

        Returns:
            torch.Tensor: The parameter tensor.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> x.value.data = torch.ones(5) * 3
            >>> x.forward()
            tensor([3., 3., 3., 3., 3.], requires_grad=True)
        """
        return self.value

    def sum(self, dim=None):
        """Create a sum operation over this variable.

        Args:
            dim: Dimension to sum over (None for all dimensions).

        Returns:
            UnaryOpExpression: Expression computing the sum.

        Examples:
            >>> x = Variable((3, 4,), name='matrix')
            >>> x.value.data = torch.ones(3, 4)
            >>> x.sum().forward()
            tensor(12., grad_fn=<SumBackward0>)
            >>> x.sum(dim=0).forward().shape
            torch.Size([4])
        """
        return UnaryOpExpression(self, lambda t: torch.sum(t, dim=dim))

    def transpose(self):
        """Create a transpose operation (for 2D variables).

        For 1D variables, returns self. For higher-dimensional variables,
        transposes the last two dimensions.

        Returns:
            Variable or UnaryOpExpression: Transposed variable.

        Examples:
            >>> A = Variable((3, 4), name='A')
            >>> A_T = A.transpose()
            >>> A.value.data = torch.randn(3, 4)
            >>> A_T.forward().shape
            torch.Size([4, 3])

            >>> x = Variable((5,) name='x')
            >>> x.transpose() is x
            True
        """
        if self.value.ndim == 1:
            return self
        return UnaryOpExpression(self, lambda t: t.transpose(-2, -1))

    @property
    def T(self):
        """Transpose property (shorthand for transpose()).

        Returns:
            Variable or UnaryOpExpression: Transposed variable.

        Examples:
            >>> A = Variable((3, 4), name='A')
            >>> A.T.forward().shape
            torch.Size([4, 3])
        """
        return self.transpose()
