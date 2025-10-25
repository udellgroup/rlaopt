"""Identity preconditioner and configuration."""

import torch

from rlaopt.linalg.preconditioners.preconditioner import (
    Preconditioner,
    PreconditionerConfig,
)


class IdentityConfig(PreconditionerConfig):
    """Configuration for the Identity preconditioner."""

    pass


class Identity(Preconditioner):
    """Identity preconditioner implementation."""

    def __init__(self, config: IdentityConfig):
        """Initialize the Identity preconditioner with the given configuration.

        Args:
            config (IdentityConfig): Configuration for the Identity preconditioner.
        """
        super().__init__(config)

    def _update(self, A: torch.Tensor):
        """Update the Identity preconditioner (no-op)."""
        pass

    def _matmul_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Identity preconditioner to the input tensor x."""
        return x

    def _inverse_matmul_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of the Identity preconditioner to the input tensor x."""
        return x
