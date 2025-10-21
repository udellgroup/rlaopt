"""Abstract base classes for preconditioners."""

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, ConfigDict


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
