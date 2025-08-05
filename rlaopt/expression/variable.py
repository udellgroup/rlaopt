"""Variable class that extends torch.nn.Parameter.

Inspired by the Variable class in cvxpy.
"""

import cvxpy as cp
import torch

from rlaopt.settings import VAR_PREFIX
from rlaopt.utils.counter import get_id
from .expression import Expression


class Variable(Expression):
    """A variable class that extends torch.nn.Parameter."""

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


        self.value = torch.nn.Parameter(
            data, requires_grad
        )
        self._set_id_and_name(var_id, name)
        # return instance

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
        return cp.Variable(shape=self.shape, name=self.name, var_id=self.id)

    def __repr__(self):
        """Full representation of the Variable."""
        info_components = [
            f"Variable(name='{self.name}'",
            f"id='{self.id}'",
            f"shape={tuple(self.shape)}",
            f"dtype={self.dtype}",
            f"device='{self.device}'",
            f"requires_grad={self.requires_grad}",
        ]
        info = ", ".join(info_components)

        return info + ")"

    def __str__(self):
        """Shortened representation of the Variable."""
        return f"Variable '{self.name}' with shape {tuple(self.shape)}"

    def is_smooth(self):
        return True

    def is_proxable(self):
        return False

    def evaluate_at(self, **variable_locations):
        if len(variable_locations) == 0:
            return self.value
        else:
            return variable_locations[self.name]

