"""Module for Variable class."""

import torch
from typing_extensions import Self

from rlaopt.expression.expression import Expression
from rlaopt.expression.tree import ExprTree
from rlaopt.settings import VAR_PREFIX
from rlaopt.utils.counter import get_id


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
    """

    def __init__(
        self,
        size_or_tensor: int | tuple[int, ...] | torch.Tensor,
        requires_grad: bool = True,
        var_id: int | None = None,
        name: str | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Constructor method for Variable."""
        super().__init__()

        if isinstance(size_or_tensor, torch.Tensor):
            data = size_or_tensor
        elif isinstance(size_or_tensor, (tuple, int)):  # Accept int or tuple
            # Normalize int to tuple
            size = (
                (size_or_tensor,) if isinstance(size_or_tensor, int) else size_or_tensor
            )
            data = torch.zeros(size, dtype=dtype, device=device)
        else:
            raise TypeError(
                f"size must be int, tuple[int, ...] or torch.Tensor, "
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

    @classmethod
    def like(cls, expr: Expression) -> Self:
        """Create a new Variable with same shape, dtype, and device as an expression.

        This is useful for creating auxiliary variables in decomposition methods
        that match the properties of an existing expression.

        Args:
            expr: Expression to match properties from.

        Returns:
            Self: New variable with same shape, dtype, and device as expr's value.
        """
        value = expr.forward()
        return cls(value.shape, dtype=value.dtype, device=value.device)

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

    def __repr__(self):
        """Full representation of the Variable.

        Returns:
            str: Detailed string representation including all attributes.
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
        """
        return f"Variable '{self.name}' with shape {self.value.shape}"

    def is_smooth(self):
        """Variables are smooth (identity function is differentiable).

        Returns:
            bool: Always True.
        """
        return True

    def forward(self) -> torch.Tensor:
        """Evaluate the variable (returns its current value).

        Returns:
            torch.Tensor: The parameter tensor.
        """
        return self.value

    def tree(self) -> ExprTree:
        """Return tree representation for Variable (leaf node).

        Returns:
            ExprTree: Leaf node with type 'Variable'.
        """
        return ExprTree(f"Variable({self._name})")

    def is_affine(self) -> bool:
        """Variables are affine expressions.

        Returns:
            bool: Always True.
        """
        return True
