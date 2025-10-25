"""Factory for creating preconditioners based on configuration."""

import torch

from rlaopt.linalg.preconditioners.identity import Identity, IdentityConfig
from rlaopt.linalg.preconditioners.nystrom import Nystrom, NystromConfig
from rlaopt.linalg.preconditioners.preconditioner import Preconditioner


def _get_preconditioner_class(
    config: IdentityConfig | NystromConfig,
) -> type[Preconditioner]:
    if isinstance(config, IdentityConfig):
        return Identity
    elif isinstance(config, NystromConfig):
        return Nystrom
    else:
        raise TypeError(f"Unknown preconditioner config type: {type(config)}")


def get_preconditioner(
    config: IdentityConfig | NystromConfig, A: torch.Tensor
) -> Preconditioner:
    """Factory function to create a preconditioner based on the given configuration.

    Args:
        config (IdentityConfig | NystromConfig): Configuration for the preconditioner.
        A (torch.Tensor): The matrix for which the preconditioner is to be created.

    Returns:
        Preconditioner: An instance of the specified preconditioner.

    Raises:
        TypeError: If the configuration type is unknown.

    """
    preconditioner_class = _get_preconditioner_class(config)
    preconditioner = preconditioner_class(config)
    preconditioner._update(A)
    return preconditioner
