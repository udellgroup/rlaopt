"""Abstract base classes for preconditioners and configurations."""

from __future__ import annotations

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

    @abstractmethod
    def _matmul_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the preconditioner to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor to which the preconditioner is applied.

        Returns:
            torch.Tensor: Result of applying the preconditioner to x.
        """
        pass

    @abstractmethod
    def _inverse_matmul_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of the preconditioner to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor to which the inverse preconditioner
            is applied.

        Returns:
            torch.Tensor: Result of applying the inverse preconditioner to x.
        """
        pass

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Overload the matrix multiplication operator to apply preconditioner.

        Args:
            x (torch.Tensor): Input tensor to which the preconditioner is applied.

        Returns:
            torch.Tensor: Result of applying the preconditioner to x.
        """
        _is_torch_tensor_1d_2d(x)
        return self._matmul_impl(x)

    def _inverse_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of the preconditioner to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor to which the inverse preconditioner
            is applied.

        Returns:
            torch.Tensor: Result of applying the inverse preconditioner to x.
        """
        _is_torch_tensor_1d_2d(x)
        return self._inverse_matmul_impl(x)

    @property
    def inv(self) -> InvPreconditioner:
        """Get the inverse preconditioner.

        Returns:
            InvPreconditioner: The inverse preconditioner.
        """
        return InvPreconditioner(self)


class InvPreconditioner:
    """Helper class to access the inverse of a preconditioner.

    This class wraps a Preconditioner instance and provides access to its
    inverse matrix multiplication method.

    Attributes:
        preconditioner (Preconditioner): The preconditioner instance.
    """

    def __init__(self, preconditioner: Preconditioner):
        """Initialize the InvPreconditioner with the given preconditioner.

        Args:
            preconditioner (Preconditioner): The preconditioner instance.
        """
        self.preconditioner = preconditioner

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Overload the matrix multiplication operator to apply inverse preconditioner.

        Args:
            x (torch.Tensor): Input tensor to which the
                inverse preconditioner is applied.

        Returns:
            torch.Tensor: Result of applying the inverse preconditioner to x.
        """
        return self.preconditioner._inverse_matmul(x)
