"""This module implements the Identity preconditioner."""

import torch

from .preconditioner import Preconditioner
from .configs import IdentityConfig


class Identity(Preconditioner):
    """Identity preconditioner class.

    This class implements the Identity preconditioner, which leaves the input unchanged.
    It's often used as a baseline or when no preconditioning is desired.

    Example:
        >>> import torch
        >>> from rlaopt.preconditioners.configs import IdentityConfig
        >>>
        >>> # Configure and initialize the Identity preconditioner
        >>> config = IdentityConfig()
        >>> preconditioner = Identity(config)
        >>>
        >>> # Use the preconditioner
        >>> x = torch.randn(100)
        >>> preconditioned_x = preconditioner @ x
        >>> assert torch.allclose(x, preconditioned_x)
        >>>
        >>> # Use the inverse preconditioner (same as forward for Identity)
        >>> inv_preconditioned_x = preconditioner._inv @ x
        >>> assert torch.allclose(x, inv_preconditioned_x)
    """

    def __init__(self, config: IdentityConfig):
        """Initialize the Identity preconditioner.

        Args:
            config (IdentityConfig): Configuration for the Identity preconditioner.
                                     This is typically empty for
                                     the Identity preconditioner.
        """
        super().__init__(config)

    def _update(self, A, device):
        """Update the Identity preconditioner.

        This method is a no-op for the Identity
        preconditioner as it doesn't require any updates.

        Args:
            A: The matrix to be preconditioned (ignored for Identity).
            device (torch.device): The device on which to perform
            computations (ignored for Identity).
        """
        pass

    def __matmul__(self, x):
        """Perform matrix multiplication with the preconditioner.

        For the Identity preconditioner,
        this operation simply returns the input unchanged.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The input tensor x, unchanged.
        """
        return x

    def _inverse_matmul_1d(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _inverse_matmul_2d(self, x: torch.Tensor) -> torch.Tensor:
        return self._inverse_matmul_1d(x)
