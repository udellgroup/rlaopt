"""Expression module for symbolic optimization modeling.

This module provides the core expression class for building symbolic mathematical
expressions that can be evaluated, differentiated, and optimized. Expressions are
built compositionally through operator overloading and can represent optimization
objectives, constraints, and models.

Classes:
    Expression: Abstract base class for all expressions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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

    def evaluate(self, params: TensorDict) -> torch.Tensor:
        """Evaluate the expression at specified parameter values.

        Unlike forward(), this method evaluates the expression at parameter
        values different from those currently stored, without modifying the
        stored parameters. Useful for line searches, parameter exploration, etc.

        Args:
            params: Dictionary mapping parameter names to their values.

        Returns:
            torch.Tensor: The evaluated result.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> x.value.data = torch.zeros(5)
            >>> result = x.evaluate({'exprs.0.x': torch.ones(5)})
            >>> torch.equal(result, torch.ones(5))
            True
        """
        return torch.func.functional_call(
            self, params.to_dict(), args=None, kwargs=None
        )

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

    def params_names(self) -> list[str]:
        """Returns the list of parameter names in order."""
        return list(self.params.keys())

    def params_shapes(self) -> list[tuple[int]]:
        """Returns a list of parameter shapes in order."""
        return [param.shape for param in self.params.values()]

    def params_from_tensors(
        self, tensors: list[torch.Tensor] | tuple[torch.Tensor]
    ) -> TensorDict:
        """Creates a params TensorDict from a list/tuple of tensors.

        Constructs a params TensorDict that match the expression's parameter structure.
        The provided tensors are mapped to parameter
        names in order. Shapes and devices are validated to see
        that they match the expression's TensorDict, which
        stores expression's parameter configuration.

        Args:
            tensors: List or tuple of tensors to convert into a TensorDict.
                    Must match the number, order, shapes, and device of
                    the expression's parameters.

        Returns:
            TensorDict with tensors mapped to appropriate parameter names,
            preserving the batch_size from self.params.

        Raises:
            ValueError: If the number of tensors doesn't match the number of parameters.
            ValueError: If any tensor shape doesn't match the expected parameter shape.
            ValueError: If tensor devices don't match the params TensorDict device.

        Example:
            >>> expr = SumSquares(Variable(shape=(3,)))
            >>> tensors = [torch.tensor([1.0, 2.0, 3.0])]
            >>> params = expr.params_from_tensors(tensors)
        """
        names = self.params_names()

        # Validate number of tensors
        if len(tensors) != len(names):
            raise ValueError(
                f"Input tensors do not define valid parameter configuration: "
                f"number of tensors ({len(tensors)}) does not match number "
                f"of parameters ({len(names)})"
            )

        dict_of_params = {}
        has_device = self.params.device is not None

        for idx, (tensor, name) in enumerate(zip(tensors, names)):
            # Validate shape
            if tensor.shape != self.params[name].shape:
                raise ValueError(
                    f"Input tensors do not define valid parameter configuration: "
                    f"shape of tensor at position {idx} is {tensor.shape}, "
                    f"expected {self.params[name].shape}. "
                    f"Call params_shapes() to get the correct shapes "
                    f"for all parameters."
                )

            # Validate device if applicable
            if has_device and tensor.device != self.params.device:
                raise ValueError(
                    f"Input tensors do not define valid parameter configuration: "
                    f"device of tensor at position {idx} is {tensor.device}, "
                    f"expected {self.params.device}"
                )

            dict_of_params[name] = tensor

        return TensorDict(dict_of_params, batch_size=self.params.batch_size)

    def update_params(self, params_dict: TensorDict):
        """Update parameters from a TensorDict.

        Args:
            params_dict (TensorDict): Dictionary mapping parameter names to new values.

        Examples:
            >>> x = Variable((5,), name='x')
            >>> x.update_params({'x': torch.ones(5)})
            >>> torch.equal(x.value, torch.ones(5))
            True
        """
        self.load_state_dict(params_dict.to_dict(), strict=False)

    def expr_convert_params(self, expr_params: TensorDict) -> TensorDict:
        """Convert parameter names from another expression to match this one.

        Given parameters from another Expression whose leaf tensor shapes match
        but have different names, returns a dictionary with names consistent
        with the current expression. Useful for transferring parameters between
        similar expressions.

        Args:
            expr_params (TensorDict): Parameters from another expression.

        Returns:
            TensorDict: Parameters with names matching this expression.
        """
        return self.params.convert_target(expr_params)

    @property
    def params(self) -> TensorDict:
        """Get parameters as a dictionary.

        Returns:
            TensorDict: Dictionary of parameter names to parameter tensors.
        """
        return TensorDict(self.params_dict())

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
