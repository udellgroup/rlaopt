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
        var_id: int | None = None,
        name: str | None = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        """Create and initialize a new Variable instance."""
        # Process size to get proper shape
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = size[0]
        else:
            shape = size

        # Create tensor
        data = torch.zeros(shape, dtype=dtype, device=device)

        # Create Parameter instance
        instance = super().__new__(cls, data, requires_grad)

        # Initialize Variable-specific attributes
        Variable._set_id_and_name(instance, var_id, name)

        return instance

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

        # Create Parameter instance
        instance = super().__new__(cls, tensor.clone(), requires_grad)

        # Initialize Variable-specific attributes
        Variable._set_id_and_name(instance, var_id, name)

        return instance

    @staticmethod
    def _set_id_and_name(instance, var_id=None, name=None):
        """Helper method to set ID and name attributes."""
        # Set ID
        if var_id is None:
            instance._id = get_id()
        else:
            instance._id = var_id

        # Set name
        if name is None:
            instance._name = f"{VAR_PREFIX}{instance._id}"
        elif isinstance(name, str):
            instance._name = name
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
