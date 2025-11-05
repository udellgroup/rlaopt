"""Expression module for symbolic optimization modeling.

This module provides the core expression class for building symbolic mathematical
expressions that can be evaluated, differentiated, and optimized. Expressions are
built compositionally through operator overloading and can represent optimization
objectives, constraints, and models.

Classes:
    Expression: Abstract base class for all expressions.
"""

from abc import ABC, abstractmethod
from collections import defaultdict

import torch

from rlaopt.expression import expr_types
from rlaopt.ext_tensordict import TensorDict


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
    def forward(self) -> torch.Tensor:
        """Evaluate the expression using current parameter values.

        This method evaluates the expression with the current values of all
        registered parameters.

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

    def evaluate(self, variables_dict: TensorDict) -> torch.Tensor:
        """Evaluate the expression at specified variable values.

        Unlike forward(), this method evaluates the expression at variable
        values different from those currently stored. Useful for line searches,
        parameter exploration, etc.

        Args:
            variables_dict: Dictionary mapping variable names to their values.

        Returns:
            torch.Tensor: The evaluated result.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> new_params = TensorDict({'x': torch.ones(5)})
            >>> result = x.evaluate(new_params)
            >>> torch.equal(result, new_params['x'])
            True
        """
        params = self._variables_dict_to_params_dict(variables_dict)

        # Save current variable values to restore after functional_call
        # This is necessary because functional_call can mutate parameters
        # when the same Variable module appears multiple times in the
        # expression tree (tied weights)
        saved_variables = TensorDict(
            {
                var_name: tensor.clone()
                for var_name, tensor in self.variables_dict.items()
            }
        )

        try:
            result = torch.func.functional_call(
                self, params, args=None, kwargs=None, tie_weights=False
            )
        finally:
            # Always restore variables, even if functional_call raises an exception
            self.update_variables(saved_variables)

        return result

    def get_variable_names(self) -> list[str]:
        """Returns the list of variable names in order."""
        return list(self.variables_dict.keys())

    def get_variable_shapes(self) -> list[tuple[int]]:
        """Returns a list of variable shapes in order."""
        return [var.shape for var in self.variables_dict.values()]

    def update_variables(self, variables_dict: TensorDict):
        """Update variables from a TensorDict.

        Args:
            variables_dict (TensorDict): TensorDict with new variable values.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> x.update_variables(TensorDict({'x': torch.ones(5)}))
            >>> torch.equal(x.value, torch.ones(5))
            True
        """
        params_dict = self._variables_dict_to_params_dict(variables_dict)
        # Use strict=False to allow partial updates
        self.load_state_dict(params_dict, strict=False)

    def _variables_dict_to_params_dict(self, variables_dict: TensorDict) -> dict:
        """Convert a variables dict to a parameters dict.

        Maps variable names to their corresponding parameter names in the
        module hierarchy.
        """
        vars_to_param_names_map = self._get_variables_to_param_names_mapping()
        params_dict = {}
        for var_name, tensor in variables_dict.items():
            for param_name in vars_to_param_names_map[var_name]:
                params_dict[param_name] = tensor
        return params_dict

    def _get_variables_to_param_names_mapping(self) -> dict[str, list[str]]:
        mapping = defaultdict(list)
        for module_path, module in self.named_modules():
            if isinstance(module, expr_types.variable()):
                # We have to account for the full module path to get
                # the correct parameter name
                full_param_name = (
                    f"{module_path}.{module._name}" if module_path else module._name
                )
                mapping[module.name].append(full_param_name)
        return mapping

    @property
    def variables_dict(self) -> TensorDict:
        """Get variables as a dictionary.

        Returns:
            TensorDict: Dictionary of variable names to variable tensors.
        """
        vars_dict = {}
        for _, module in self.named_modules():
            if (
                isinstance(module, expr_types.variable())
                and module.name not in vars_dict
            ):
                vars_dict[module.name] = module.value
        return TensorDict(vars_dict)

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
        return expr_types.add_expr()(self, other)

    def __radd__(self, other):
        """Add a scalar and an expression (reverse operation).

        Args:
            other: Float or int to add.

        Returns:
            AddExpression: Sum of other and self.
        """
        return expr_types.add_expr()(other, self)

    def __sub__(self, other):
        """Subtract an expression or scalar from this expression.

        Args:
            other: Expression, float, or int to subtract.

        Returns:
            AddExpression: Difference of self and other.
        """
        return expr_types.add_expr()(self, -other)

    def __rsub__(self, other):
        """Subtract this expression from a scalar (reverse operation).

        Args:
            other: Float or int to subtract from.

        Returns:
            AddExpression: Difference of other and self.
        """
        return expr_types.add_expr()(other, -self)

    def __neg__(self):
        """Negate this expression.

        Returns:
            ProductExpression: Negation of self.
        """
        return expr_types.prod_expr()(expr_types.constant()(-1.0), self)

    def __mul__(self, other):
        """Multiply this expression by another (elementwise).

        Args:
            other: Expression, float, or int to multiply.

        Returns:
            ProductExpression: Elementwise product of self and other.
        """
        return expr_types.prod_expr()(self, other, matmul=False)

    def __rmul__(self, other):
        """Multiply a scalar by this expression (reverse operation).

        Args:
            other: Float or int to multiply.

        Returns:
            ProductExpression: Elementwise product of other and self.
        """
        return expr_types.prod_expr()(other, self, matmul=False)

    def __truediv__(self, other):
        """Divide this expression by a scalar.

        Args:
            other: Float or int to divide by.

        Returns:
            ProductExpression: Result of division.
            NotImplemented: If other is not a scalar.
        """
        if isinstance(other, (int, float)):
            return expr_types.prod_expr()(
                self, expr_types.constant()(1.0 / other), matmul=False
            )
        return NotImplemented

    def __matmul__(self, other):
        """Matrix multiply this expression by another.

        Args:
            other: Expression, float, or int to multiply.

        Returns:
            ProductExpression: Matrix product of self and other.
        """
        return expr_types.prod_expr()(self, other, matmul=True)

    def __rmatmul__(self, other):
        """Matrix multiply a value by this expression (reverse operation).

        Args:
            other: Expression, float, or int to multiply.

        Returns:
            ProductExpression: Matrix product of other and self.
        """
        return expr_types.prod_expr()(other, self, matmul=True)

    def __pow__(self, exponent):
        """Raise this expression to a power (elementwise).

        Args:
            exponent: Power to raise to.

        Returns:
            UnaryOpExpression: Result of exponentiation.
        """
        return expr_types.unary_op_expr()(self, lambda t: torch.pow(t, exponent))
