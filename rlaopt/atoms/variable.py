"""Variable class that extends torch.nn.Parameter.

Inspired by the Variable class in cvxpy.
"""

import torch

from rlaopt.settings import VAR_PREFIX
from rlaopt.utils.counter import get_id


class Variable(torch.nn.Parameter):
    """A variable atom that extends torch.nn.Parameter."""

    def __init__(
        self,
        data: torch.Tensor,
        requires_grad: bool = True,
        var_id: int | None = None,
        name: str | None = None,
    ):
        """Initializes the Variable atom.

        Args:
            data: Initial data for the variable.
            requires_grad: Whether to compute gradients with respect to this variable.
            var_id: Optional identifier for the variable.
            If None, a new id is generated.
            name: Optional name for the variable.
            If None, a default name is generated using the variable id.
        """
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

        super().__init__(data, requires_grad=requires_grad)

    @property
    def id(self) -> int:
        """Returns the unique identifier of the variable."""
        return self._id

    @property
    def name(self) -> str:
        """Returns the name of the variable."""
        return self._name
