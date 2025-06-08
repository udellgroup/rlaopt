"""Variable class that extends torch.nn.Parameter.

Inspired by the Variable class in cvxpy.
"""

import torch

from rlaopt.settings import VAR_PREFIX
from rlaopt.utils.counter import get_id


class Variable(torch.nn.Parameter):
    """A variable class that extends torch.nn.Parameter."""

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        requires_grad: bool = True,
        var_id: int | None = None,
        name: str | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Initializes the Variable class with zeros of the given shape.

        Args:
            shape: Shape of the variable (int or tuple of ints).
            requires_grad: Whether to compute gradients with respect to this variable.
            var_id: Optional identifier for the variable.
                If None, a new id is generated.
            name: Optional name for the variable.
                If None, a default name is generated using the variable id.
            dtype: Data type of the variable. Default: None
            device: Device to place the variable on. Default: None
        """
        # Set id and name
        if var_id is None:
            self._id = get_id()
        else:
            self._id = var_id

        if name is None:
            self._name = f"{VAR_PREFIX}{self._id}"
        elif isinstance(name, str):
            self._name = name
        else:
            raise TypeError(f"Expected name to be a string, got {type(name)} instead.")

        # Create zeros tensor with the given shape
        data = torch.zeros(shape, dtype=dtype, device=device)

        # Initialize the Parameter with the data
        super().__init__(data, requires_grad=requires_grad)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        requires_grad: bool | None = None,
        var_id: int | None = None,
        name: str | None = None,
    ):
        """Initialize a Variable from an existing tensor.

        Args:
            tensor: Existing tensor to use as initial data.
            requires_grad: Whether to compute gradients with respect to this variable.
                If None, it defaults to the requires_grad attribute of the tensor.
            var_id: Optional identifier for the variable.
            name: Optional name for the variable.

        Returns:
            A new Variable instance initialized with the given tensor data.
        """
        if requires_grad is None:
            requires_grad = tensor.requires_grad
        variable = cls(
            tensor.shape,
            requires_grad=requires_grad,
            var_id=var_id,
            name=name,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        # Replace the zeros data with the provided tensor data
        variable.data.copy_(tensor)

        return variable

    @property
    def id(self) -> int:
        """Returns the unique identifier of the variable."""
        return self._id

    @property
    def name(self) -> str:
        """Returns the name of the variable."""
        return self._name
