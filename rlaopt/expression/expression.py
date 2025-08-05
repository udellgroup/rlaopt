from typing import Callable, Dict, Iterable
from abc import ABC, abstractmethod

import cvxpy as cp
import torch

from ..params.params import Params


class Expression(torch.nn.Module, ABC):
    """Base class for mathematical expressions in optimization problems."""

    def __init__(self):
        """Initialize the expression."""
        self._params = None
        super().__init__()

    def forward(self):
        """Evaluate the expression using registered variables.

        Returns:
            The evaluated expression
        """
        # Forward simply calls evaluate_at with no substitutions
        return self.evaluate_at()
    
    def functional_forward(self, params_dict: Dict[str, torch.Tensor])->torch.Tensor:
        return torch.func.functional_call(self, params_dict)
    
    def evaluate(self, params: Params):
        return self.functional_forward(params.value)
    
    def get_params_dict(self)->Dict[str, torch.Tensor]: 
        return dict(self.named_parameters())

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

    @property
    def params(self):
        if self._params is None:
           self._params = Params(self.get_params_dict())
        return self._params
    
    def update_params(self, params: Params):
        self.load_state_dict(params.value, strict=False)
        self._params = Params(self.get_params_dict())
    
    def expr_convert_params(self, params: Params):
        """Converts params of external model to be consistent with current model.
           
           Given params from another Expression whose leaf tensor shapes are consistent with current Expression but have different names,  
           this function returns a new Params object where each leaf tensor's name is consistent with the current expression.
        """
        if self.params.has_compatible_shapes(params):
            params_dict = {name: tensor for name, tensor in zip(self.params.names, params.data)}
        else:
            raise ValueError("params is not compatible with Expression params")
        return Params(params_dict)
    
    def params_from_tensors(self, tensors: Iterable[torch.Tensor]):
        """Converts iterable of tensors into Params object for the current Expression.

           Given iterable of tensors whose shapes are consistent with the leaf tensors of params for the 
           current expression, this function returns a new Params object where each leaf tensor is a tensor 
           in tensors.   
        """
        params_dict = {name: tensor for name, tensor in zip(self.params.names, tensors)}
        return Params(params_dict)
                            

    def __call__(self, **variable_locations):
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
        return cp.Constant(self.value.numpy())

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
        evaluated_exprs = [expr.evaluate_at(**variable_locations) for expr in self.exprs]
        evaluated_exprs = torch.stack(evaluated_exprs)
        return self.op(evaluated_exprs, torch)

    def to_cvxpy(self):
        """Convert to a CVXPY expression."""
        return self.op([expr.to_cvxpy() for expr in self.exprs], cp)


class AddExpression(BinaryOperatorExpression):
    """Expression representing addition."""
    def __init__(self, left_or_exprs, right=None):
        super().__init__(left_or_exprs, right)
        self._num_non_smooth_exprs = self._count_non_smooth_terms()
        self._prox = self._build_prox()
        
    def op(self, exprs, lib):
        return lib.sum(exprs)

    def is_proxable(self):
        # If any of non-smooth terms isn't proxable, then the sum isn't proxable
        if any(not expr.is_proxable() and not expr.is_smooth() for expr in self.exprs):
            return False
        # Non-smooth terms must all be proxable, if there is only one
        # then the resulting AddExpression must be Proxable
        elif self.num_non_smooth_exprs == 1:
            return True
        # If the non-smooth terms have no overlap, the resulting AddExpression is proxable
        # otherwise it isn't
        else:
            length = 0
            params = set()
            for expr in self.exprs:
                expr_params = list(expr.parameters())
                length += len(expr_params)
                params.update(expr_params)
                if length != len(params):
                    return False
            return True
    
    def prox(self, location, prox_scaling):
        return self._prox(location, prox_scaling)

    def operator_split(self):
        """
        Splits sum of operators into smooth and non-smooth part
        """
        smooth = []
        non_smooth = []
        for expr in self.exprs:
            if expr.is_smooth():
                smooth.append(expr)
            else:
                non_smooth.append(expr)
        if non_smooth:
            non_smooth_expr = AddExpression(non_smooth)
        else:
            non_smooth_expr = None
        return AddExpression(smooth), non_smooth_expr
    
    @property
    def num_non_smooth_exprs(self):
        return self._num_non_smooth_exprs
    
    def _build_prox(self):
        """
        Builds the proximal operator for AddExpression.
        """
        # If not proxable, prox should raise not implemented error
        if not self.is_proxable():
            def prox(location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
                raise NotImplementedError("Expression is not proxable")
            return prox
        # If there is only one non-smooth proxable expr, AddExpression is proxable
        elif self._num_non_smooth_exprs == 1:
            proxes = self._get_proxes()
            return proxes[0]
        # Otherwise we have sum of prox of terms with no overlap, so apply prox of each term to 
        # appropriate param group
        else:
            proxes = self._get_proxes()
            def prox(location: Dict[str, torch.Tensor], prox_scaling: float)-> Dict[str, torch.Tensor]:
                return {name : prox_fn(loc, prox_scaling) for (name, loc), prox_fn in zip(location, proxes)}
        return prox

    def _count_non_smooth_terms(self):
        count = 0
        for expr in self.exprs:
            if not expr.is_smooth():
                count += 1
        return count
    
    def _get_proxes(self) -> list[Callable[[torch.Tensor, float], torch.Tensor]]:
        """
        Returns list of proxes of non-smooth expressions
        """
        proxes = []
        for expr in self.exprs:
            if not expr.is_smooth():
                proxes.append(expr.prox)        
        return proxes
    
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