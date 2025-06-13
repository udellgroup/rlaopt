from abc import ABC, abstractmethod
import cvxpy as cp
import torch


class Expression(torch.nn.Module, ABC):
    """Base class for mathematical expressions in optimization problems."""

    def __init__(self):
        """Initialize the expression."""
        super().__init__()

    def forward(self):
        """Evaluate the expression using registered variables.

        Returns:
            The evaluated expression
        """
        # Forward simply calls evaluate_at with no substitutions
        return self.evaluate_at()

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

    def __call__(self, **variable_locations):
        """Call the expression, choosing the appropriate evaluation method.

        If called with no arguments, uses forward(). If called with keyword arguments,
        uses evaluate_at().
        """
        if not variable_locations:
            return self.forward()
        else:
            return self.evaluate_at(**variable_locations)

    def __add__(self, other):
        return AddExpression(self, other)

    def __radd__(self, other):
        return AddExpression(other, self)

    def __mul__(self, other):
        left = self
        right = to_expr(other)
        if left.parameters() == 0:
            const = left
            nonconst = right
        else:
            const = right
            nonconst = left
        if isinstance(nonconst, AddExpression):
            return AddExpression([const * expr for expr in nonconst.exprs])
        else:
            return MulExpression(left, right)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return MulExpression(-1.0, self)

    def __sub__(self, other):
        return AddExpression(self, -other)

    def __rsub__(self, other):
        return AddExpression(other, -self)

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Proximal operator of the atom.

        This method should only be called if the atom is proxable. Otherwise, it should
        raise a NotImplementedError.

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Scaling factor for the proximal operator

        Returns:
            Result of the proximal operator
        """
        raise NotImplementedError


class ConstExpression(Expression):
    def __init__(self, value):
        super().__init__()
        if isinstance(value, torch.Tensor):
            self.value = torch.nn.parameter.Buffer(value)
        else:
            self.value = torch.nn.parameter.Buffer(torch.tensor(value))

    def is_smooth(self) -> bool:
        return  True

    def is_proxable(self) -> bool:
        return  True

    def evaluate_at(self, **variable_locations):
        return self.value

    def to_cvxpy(self) -> cp.Expression:
        return cp.Constant(value.numpy())

def to_expr(val):
    if isinstance(val, Expression):
        return val
    else:
        return ConstExpression(val)


class BinaryOperatorExpression(Expression, ABC):
    """Expression representing addition."""
    
    @abstractmethod
    def op(self, exprs, lib):
        pass
    
    def validate(self):
        pass

    def __init__(self, left_or_exprs, right=None):
        """
        Constructors are either 
            left: Expression, right: Expression
        or
            exprs: list[Expression]
        """

        super().__init__()
        if right is not None:
            exprs = []
            left = to_expr(left_or_exprs)
            right = to_expr(right)
            if isinstance(left, type(self)):
                exprs.extend(left.exprs)
            else:
                exprs.append(left)

            if isinstance(right, type(self)):
                exprs.extend(right.exprs)
            else:
                exprs.append(right)
        else:
            exprs = left_or_exprs

        self.exprs = torch.nn.ModuleList(exprs)

        self.validate()

    def is_smooth(self) -> bool:
        """Check if the addition is smooth."""
        # Both left and right must be smooth for the addition to be smooth
        return all([expr.is_smooth() for expr in self.exprs])

    def evaluate_at(self, **variable_locations):
        """Evaluate the addition at specific variable locations."""
        return self.op([expr.evaluate_at(**variable_locations) for expr in self.exprs], torch)

    def to_cvxpy(self):
        """Convert to a CVXPY expression."""
        return self.op([expr.to_cvxpy() for expr in self.exprs], cvxpy)


class AddExpression(BinaryOperatorExpression):
    """Expression representing addition."""

    def op(self, exprs, lib):
        return lib.sum(exprs)

    def is_proxable(self):
        if any(not expr.is_proxable() for expr in self.exprs):
            return False
        length = 0
        params = set()
        for expr in exprs:
            expr_params = list(expr.parameters())
            length += len(expr_params)
            params.extend(expr_params)
            if length != len(params):
                return False
        return True

    def operator_split(self):
        """
        Splits sum of operators into smooth and proxable part
        """
        smooth = []
        prox = []
        for expr in exprs:
            if expr.is_smooth():
                smooth.append(expr)
            else:
                prox.append(expr)
        prox_expr = AddExpression(prox)
        if not prox_expr.is_proxable():
            raise ValueError("Cannot split operator")
        return AddExpression(smooth), prox_expr



class MulExpression(Expression):
    """Expression representing multiplication."""

    def op(self, exprs, lib):
        return lib.prod(exprs)

    def validate(self):
        if sum(len(expr.parameters()) != 0 for expr in self.exprs) > 1:
            raise TypeError("Cannot multiply two nonconstant Expressions.")

    def is_proxable(self):
        """Assumes that this is a const scalar times a function"""
        for expr in self.exprs:
            if len(expr.parameters()) != 0:
                return expr.is_proxable()
        return True
