"""Variable class that extends torch.nn.Parameter.

Inspired by the Variable class in cvxpy.
"""

import torch

from rlaopt.settings import VAR_PREFIX
from rlaopt.utils.counter import get_id


class Variable(torch.nn.Parameter):
    """A variable class that extends torch.nn.Parameter."""

    def __new__(
        cls,
        *size,
        requires_grad: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        **kwargs,
    ):
        """Create a new Variable instance.

        Args:
            *size: Shape dimensions (as individual integers or a tuple)
            requires_grad: Whether to track gradients for this variable
            dtype: Data type of the tensor
            device: Device to place the tensor on
            **kwargs: Additional keyword arguments captured for __init__ later
        """
        # Process size to get proper shape
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = size[0]
        else:
            shape = size

        # Create tensor first
        data = torch.zeros(shape, dtype=dtype, device=device)

        # Call Parameter's __new__ with tensor
        return super().__new__(cls, data, requires_grad)

    def __init__(
        self, *args, var_id: int | None = None, name: str | None = None, **kwargs
    ):
        """Initialize Variable attributes.

        Args:
            *args: Positional arguments (not used, kept for compatibility)
            var_id: Optional identifier for the variable
            name: Optional name for the variable
            **kwargs: Additional keyword arguments (not used, kept for compatibility)
        """
        # No need to call super().__init__

        # Set id
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
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        requires_grad: bool | None = None,
        var_id: int | None = None,
        name: str | None = None,
    ):
        """Create a Variable from an existing tensor."""
        if requires_grad is None:
            requires_grad = tensor.requires_grad

        # Create Parameter first
        param = super().__new__(cls, tensor.clone(), requires_grad)

        # Set Variable-specific attributes
        if var_id is None:
            param._id = get_id()
        else:
            param._id = var_id

        if name is None:
            param._name = f"{VAR_PREFIX}{param._id}"
        elif isinstance(name, str):
            param._name = name
        else:
            raise TypeError(f"Expected name to be a string, got {type(name)} instead.")

        return param

    @property
    def id(self) -> int:
        """Returns the unique identifier of the variable."""
        return self._id

    @property
    def name(self) -> str:
        """Returns the name of the variable."""
        return self._name
