from abc import ABC, abstractmethod
from functools import reduce
from typing import Callable, Dict, Iterable, Union


import cvxpy as cp
import torch

from rlaopt.settings import VAR_PREFIX
from rlaopt.utils.counter import get_id
from rlaopt.utils import tensor_dict_ops as dict_ops
from rlaopt._typing import TensorDict


# ===============================
# Base Expression
# ===============================
class Expression(torch.nn.Module, ABC):
    """Base class for all expressions."""

    def __init__(self):
        super().__init__()

    def forward(self):
        """Evaluate the expression using registered variables.

        Returns:
            The evaluated expression
        """
        # Forward simply calls evaluate_at with no substitutions
        return self.evaluate_at()

    def functional_forward(self, params: TensorDict) -> torch.Tensor:
        return torch.func.functional_call(self, params)

    def evaluate(self, params: TensorDict):
        return self.functional_forward(params)

    def params_dict(self) -> TensorDict:
        return dict(self.named_parameters())
    
    def update_params(self, params_dict: TensorDict):
        """Update parameters from a dictionary."""
        self.load_state_dict(params_dict, strict=False)

    def expr_convert_params(
        self, params_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Converts params of external model to be consistent with current model.

        Given params from another Expression whose leaf tensor shapes are consistent
        with current Expression but have different names, this function returns a new
        dict where each leaf tensor's name is consistent with the current expression.
        
        """
        return dict_ops.relabel_from_template(params_dict, self.params)
    
    @property
    def params(self) -> TensorDict:
        """Get parameters as a dictionary."""
        return self.params_dict()
    
    def __call__(self, **variable_locations):
        return self.evaluate_at(**variable_locations)

    @abstractmethod
    def is_smooth(self) -> bool:
        """Check if the expression is smooth (differentiable everywhere).

        Returns:
            True if the expression is smooth, False otherwise
        """
        pass

    @abstractmethod
    def is_proxable(self) -> bool:
        """Check if the expression is proxable.

        Returns:
            True if the expression is proxable, False otherwise
        """
        pass

    @abstractmethod
    def evaluate_at(self, **variable_locations):
        """Evaluate the expression at specific variable locations.

        Args:
            **variable_locations: Mapping of variable names to locations

        Returns:
            The evaluated expression with variables substituted with their locations
        """
        pass

    @abstractmethod
    def to_cvxpy(self) -> cp.Expression:
        """Convert the expression to a CVXPY expression.

        Returns:
            A CVXPY expression representing this mathematical expression.
        """
        pass

    # ----------------------
    # Centralized operator overloads
    # ----------------------
    def __add__(self, other):
        return AddExpression(self, other)

    def __radd__(self, other):
        return AddExpression(other, self)

    def __sub__(self, other):
        return AddExpression(self, -other)

    def __rsub__(self, other):
        return AddExpression(other, -self)

    def __neg__(self):
        return ProductExpression(ConstExpression(-1.0), self)

    def __mul__(self, other):
        return ProductExpression(self, other, matmul=False)

    def __rmul__(self, other):
        return ProductExpression(other, self, matmul=False)

    def __truediv__(self, other):
        # treat division as multiplication by reciprocal scalar where appropriate
        if isinstance(other, (int, float)):
            return ProductExpression(self, ConstExpression(1.0 / other), matmul=False)
        return NotImplemented

    def __matmul__(self, other):
        return ProductExpression(self, other, matmul=True)

    def __rmatmul__(self, other):
        return ProductExpression(other, self, matmul=True)

    def __pow__(self, exponent):
        return UnaryOpExpression(self, lambda t: torch.pow(t, exponent))

# ===============================
# Constants
# ===============================
class ConstExpression(Expression):
    def __init__(self, value: Union[float, int, torch.Tensor]):
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        # register as buffer so it is visible but not a parameter
        self.register_buffer("_value", value)

    @property
    def value(self):
        return getattr(self, "_value")

    def is_smooth(self) -> bool:
        return True

    def is_proxable(self) -> bool:
        return True

    def evaluate_at(self, **variable_locations):
        return self.value

    def to_cvxpy(self) -> cp.Expression:
        return cp.Constant(self.value.detach().cpu().numpy())

    def __neg__(self):
        # keep as constant
        return ConstExpression(-self.value)


class NAryOperatorExpression(Expression, ABC):
    """Base class for n-ary operations with automatic flattening of same-type ops."""

    @abstractmethod
    def op(self, exprs: list[torch.Tensor], lib):
        """Apply the operator to a list of evaluated tensors using `lib` (torch/cp)."""
        pass

    def validate(self):
        """Optional validation hook for subclasses."""
        pass

    def __init__(self, left_or_exprs, right=None):
        super().__init__()

        if right is not None:
            # Assume left_or_exprs is a single Expression or convertible
            if isinstance(left_or_exprs, Iterable) and not isinstance(left_or_exprs, (Expression, ConstExpression, float, int, torch.Tensor)):
                raise TypeError("When right is provided, left_or_exprs must be a single Expression, not an iterable")
            left_or_exprs = to_expr(left_or_exprs)
            right = to_expr(right)
            exprs = [left_or_exprs, right]
        else:
            # left_or_exprs is iterable of expressions
            if not isinstance(left_or_exprs, Iterable) or isinstance(left_or_exprs, Expression):
                raise TypeError("When right is None, left_or_exprs must be iterable of Expressions")
            exprs = [to_expr(e) for e in left_or_exprs]

        self.exprs = torch.nn.ModuleList(exprs)
        self.validate()

    def is_smooth(self) -> bool:
        return all(expr.is_smooth() for expr in self.exprs)

    def evaluate_at(self, **variable_locations):
        vals = [expr.evaluate_at(**variable_locations) for expr in self.exprs]
        return self.op(vals, torch)

    def to_cvxpy(self):
        return self.op([expr.to_cvxpy() for expr in self.exprs], cp)
    
    @abstractmethod
    def is_proxable(self) -> bool:
        pass

# ===============================
# AddExpression (sum of exprs)
# ===============================

class AddExpression(NAryOperatorExpression):
    """Sum of expressions. Accepts either two operands or a list."""

    def __init__(self, left_or_exprs, right=None):
        super().__init__(left_or_exprs, right)
        self._num_non_smooth_exprs = self._count_non_smooth_terms()
        self._prox = self._build_prox()

    def op(self, exprs, lib):
        if not exprs:
            raise ValueError("AddExpression requires at least one operand")
        if len(exprs) == 1:
            return exprs[0]
        if lib is torch:
            stacked = lib.stack(exprs, dim=0)
            return lib.sum(stacked, dim=0)
        else:
            # CVXPY: reduce using +
            return reduce(lambda a, b: a + b, exprs)
    
    def is_proxable(self):
        # If any of non-smooth terms isn't proxable, then the sum isn't proxable
        if any(not expr.is_proxable() and not expr.is_smooth() for expr in self.exprs):
            return False

        # If the non-smooth terms have no overlap,
        # then the resulting AddExpression is proxable,
        # otherwise it isn't
        length = 0
        params = set()
        for expr in self.exprs:
            expr_params = list(expr.parameters())
            length += len(expr_params)
            params.update(expr_params)
            if length != len(params):
                return False
        return True
    
    def operator_split(self):
        """Splits sum of operators into smooth and non-smooth part."""
        smooth = []
        non_smooth = []
        for expr in self.exprs:
            if expr.is_smooth():
                smooth.append(expr)
            else:
                non_smooth.append(expr)
        non_smooth_expr = AddExpression(non_smooth) if non_smooth else None
        return AddExpression(smooth) if smooth else None, non_smooth_expr

    
    def prox(self, location, prox_scaling):
        return self._prox(location, prox_scaling)

    @property
    def num_non_smooth_exprs(self):
        return self._num_non_smooth_exprs

    # TODO(pratik): figure out what is going on in this method,
    # especially in the else case
    def _build_prox(self):
        """Builds the proximal operator for AddExpression."""
        # If not proxable, prox should raise not implemented error
        if not self.is_proxable():

            def prox(location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
                raise NotImplementedError("Expression is not proxable")

            return prox
        
        # If there is only one non-smooth proxable expr, AddExpression is proxable
        elif self._num_non_smooth_exprs == 1:
            proxes = self._get_proxes()
            return proxes[0]
        
        # Otherwise we have sum of prox of terms with no overlap,
        # so apply prox of each term to appropriate param group
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
        return sum(1 for expr in self.exprs if not expr.is_smooth())

    def _get_proxes(self) -> list[Callable[[torch.Tensor, float], torch.Tensor]]:
        """Returns list of proxes of non-smooth expressions."""
        proxes = []
        for expr in self.exprs:
            if not expr.is_smooth():
                proxes.append(expr.prox)
        return proxes

    def to_cvxpy(self) -> cp.Expression:
        return super().to_cvxpy()


# ===============================
# ProductExpression (* and @)
# ===============================
class ProductExpression(NAryOperatorExpression):
    """N-ary product. If matmul=True, does sequential matrix-multiplication."""

    def __init__(self, left_or_exprs, right=None, matmul: bool = False):
        self.matmul = matmul
        super().__init__(left_or_exprs, right)

    def validate(self):
        # disallow arbitrary parameterized expressions multiplied together unless
        # all parameterized leaves are Variables or Constants.
        param_exprs = [e for e in self.exprs if any(True for _ in e.parameters())]
        if len(param_exprs) > 1:
            if not all(self._all_leaves_are_vars_or_consts(e) for e in param_exprs):
                raise TypeError(
                    "Cannot multiply two arbitrary parameterized Expressions. "
                    "Allowed: Variable * (@) Variable, or chained products of Variables/Consts."
                )

    def op(self, exprs, lib):
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
            # CVXPY: use @ for matmul, * for elementwise
            if self.matmul:
                return reduce(lambda a, b: a @ b, exprs)
            else:
                return reduce(lambda a, b: a * b, exprs)

    def _all_leaves_are_vars_or_consts(self, expr: Expression) -> bool:
        if isinstance(expr, (Variable, ConstExpression)):
            return True
        if isinstance(expr, ProductExpression):
            return all(self._all_leaves_are_vars_or_consts(child) for child in expr.exprs)
        if isinstance(expr, UnaryOpExpression):
            return self._all_leaves_are_vars_or_consts(expr.operand)
        if isinstance(expr, AddExpression):
            return all(self._all_leaves_are_vars_or_consts(c) for c in expr.exprs)
        # conservative default
        return False
    
    def is_smooth(self):
        return True

    def is_proxable(self):
        return False

    def to_cvxpy(self):
        return super().to_cvxpy()

# ===============================
# Unary ops
# ===============================
class UnaryOpExpression(Expression):
    def __init__(self, operand, op: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        expr = to_expr(operand)
        self.operands = torch.nn.ModuleList([expr])
        self._op = op

    @property
    def operand(self) -> Expression:
        return self.operands[0]

    def evaluate_at(self, **variable_locations):
        v = self.operand.evaluate_at(**variable_locations)
        return self._op(v)

    def is_smooth(self) -> bool:
        return self.operand.is_smooth()

    def is_proxable(self) -> bool:
        return False

    def to_cvxpy(self):
        raise NotImplementedError("to_cvxpy not implemented for UnaryOpExpression")


# # ===============================
# # Variable (leaf)
# # ===============================
class Variable(Expression):
    """
    Leaf optimization variable that actually registers a torch.nn.Parameter.
    name should be unique identifier used for evaluate_at substitutions.
    This class extends torch.nn.Parameter.
    """

    def __init__(
        self,
        *size_or_tensor,
        requires_grad: bool = True,
        var_id: int | None = None,
        name: str | None = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        """Create and initialize a new Variable instance."""
        super().__init__()
        # Process size to get proper shape
        if len(size_or_tensor) == 1 and isinstance(size_or_tensor[0], torch.Tensor):
            data = size_or_tensor[0]
        else:
            size = size_or_tensor
            if len(size) == 1 and isinstance(size[0], (tuple, list)):
                shape = size[0]
            else:
                shape = size
            # Create tensor
            data = torch.zeros(shape, dtype=dtype, device=device)

        self.value = torch.nn.Parameter(data, requires_grad)
        self._set_id_and_name(var_id, name)

    def _set_id_and_name(self, var_id=None, name=None):
        """Helper method to set ID and name attributes."""
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
    def id(self) -> int:
        """Returns the unique identifier of the variable."""
        return self._id

    @property
    def name(self) -> str:
        """Returns the name of the variable."""
        return self._name

    def to_cvxpy(self) -> cp.Variable:
        return cp.Variable(shape=self.value.shape, name=self.name, var_id=self.id)
    
    
    def __repr__(self):
        """Full representation of the Variable."""
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
        """Shortened representation of the Variable."""
        return f"Variable '{self.name}' with shape {self.value.shape}"

    def is_smooth(self):
        return True

    def is_proxable(self):
        return False

    def evaluate_at(self, **variable_locations):
        if len(variable_locations) == 0:
            return self.value
        else:
            return variable_locations[self.name]
    
    def sum(self, dim=None):
        return UnaryOpExpression(self, lambda t: torch.sum(t, dim=dim))

    def transpose(self):
        return UnaryOpExpression(self, lambda t: t.transpose(-2, -1))
    
    @property
    def T(self):
        return self.transpose()

    
# -----------------------
# Small helper
# -----------------------
def to_expr(val: Expression | float | int | torch.Tensor) -> Expression:
    if isinstance(val, Expression):
        return val
    elif isinstance(val, (float, int, torch.Tensor)):
        return ConstExpression(val)
    else:
        raise TypeError("Expected input of type Union[Expression, float, int, torch.Tensor]," \
        f"but got {type(val)}")