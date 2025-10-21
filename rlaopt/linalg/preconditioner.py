"""Abstract base classes for preconditioners."""

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, ConfigDict


def _is_torch_tensor_1d_2d(tensor: torch.Tensor):
    """Check if the input is a 1D or 2D torch tensor.

    Args:
        tensor: Input tensor to check.

    Raises:
        ValueError: If the input is not a torch tensor or not 1D/2D.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch tensor.")
        if tensor.ndim not in (1, 2):
            raise ValueError("Input tensor must be 1D or 2D.")


class PreconditionerConfig(BaseModel):
    """Base configuration class for preconditioners."""

    model_config = ConfigDict(extra="forbid")


class Preconditioner(ABC):
    """Abstract base class for preconditioners."""

    def __init__(self, config: PreconditionerConfig):
        """Initialize the preconditioner with the given configuration.

        Args:
            config (PreconditionerConfig): Configuration for the preconditioner.
        """
        self._config = config

    @abstractmethod
    def _update(self, A: torch.Tensor, device: torch.device, *args, **kwargs):
        """Update the preconditioner based on the matrix A.

        Args:
            A (torch.Tensor): The matrix for which to compute the preconditioner.
            device (torch.device): The device on which computations are performed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass
