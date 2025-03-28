"""This module defines abstract base classes for preconditioners.

It includes the Preconditioner abstract base class and a helper class for inverse
preconditioners.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

import torch

from .configs import PreconditionerConfig
from rlaopt.linops import LinOpType


__all__ = ["Preconditioner"]


class Preconditioner(ABC):
    """Abstract base class for preconditioners.

    This class defines the interface for preconditioners.

    Attributes:
        config (PreconditionerConfig): Configuration for the preconditioner.
    """

    def __init__(self, config: PreconditionerConfig):
        """Initialize the Preconditioner.

        Args:
            config (PreconditionerConfig): Configuration for the preconditioner.
        """
        self.config = config

    @abstractmethod
    def _update(
        self,
        A: Union[torch.Tensor, LinOpType],
        device: torch.device,
        *args: list,
        **kwargs: dict
    ):
        """Update the preconditioner.

        Args:
            A (Union[torch.Tensor, LinOpType]): The matrix or linear operator
            for which we are constructing a preconditioner.
            device (torch.device): The device on which to perform the update.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def __matmul__(self, x: torch.Tensor):
        """Perform matrix multiplication with the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        pass

    @abstractmethod
    def _inverse_matmul(self, x: torch.Tensor):
        """Perform matrix multiplication with the inverse of the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the inverse matrix multiplication.
        """
        pass

    def _inverse_matmul_compose(self, fn: Callable):
        """Compose the inverse of the preconditioner with a function.

        Args:
            fn (Callable): The function to compose with.

        Returns:
            Callable: The composed function.
        """

        def composed_fn(*args, **kwargs):
            return self._inverse_matmul(fn(*args, **kwargs))

        return composed_fn

    def _update_damping(self, baseline_rho: float):
        """Update the damping parameter based on the damping strategy.

        For most preconditioners, this is a no-op. The only exception is
        the Nyström preconditioner.

        Args:
            baseline_rho (float): The baseline damping parameter.
        """
        pass

    @property
    def _inv(self):
        """Get the inverse preconditioner.

        Returns:
            _InvPreconditioner: An instance of _InvPreconditioner
            for this preconditioner.
        """
        return _InvPreconditioner(self)


class _InvPreconditioner:
    """Helper class for inverse preconditioners.

    This class wraps a Preconditioner instance and provides the inverse operation.

    Attributes:
        preconditioner (Preconditioner): The original preconditioner.
    """

    def __init__(self, preconditioner: Preconditioner):
        """Initialize the _InvPreconditioner.

        Args:
            preconditioner (Preconditioner): The original preconditioner to invert.
        """
        self.preconditioner = preconditioner

    def __matmul__(self, x: torch.Tensor):
        """Perform matrix multiplication with the inverse preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the inverse matrix multiplication.
        """
        return self.preconditioner._inverse_matmul(x)
