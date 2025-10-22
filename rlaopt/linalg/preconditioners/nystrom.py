"""Nyström preconditioner and configuration."""

from typing import Literal

import torch
from pydantic import Field

from rlaopt.linalg.preconditioners.preconditioner import (
    Preconditioner,
    PreconditionerConfig,
)


class NystromConfig(PreconditionerConfig):
    """Configuration for the Nyström preconditioner."""

    # TODO(pratik): add option for sketching method

    rank: int = Field(gt=0)
    damping: float = Field(ge=0.0)
    damping_mode: Literal["adaptive", "non_adaptive"] = "adaptive"


class Nystrom(Preconditioner):
    """Nyström preconditioner implementation."""

    def __init__(self, config: NystromConfig):
        """Initialize the Nyström preconditioner with the given configuration.

        Args:
            config (NystromConfig): Configuration for the Nyström preconditioner.
        """
        super().__init__(config)
        self.U = None
        self.S = None
        self.L = None
        self.using_low_precision = False

    def _update(self, A: torch.Tensor, device: torch.device):
        """Update the Nyström preconditioner based on the matrix A."""
        if A.dtype != torch.float64:
            self.using_low_precision = True

        # Sketching matrix
        Omega = _generate_ortho_embedding(
            dimension=A.shape[0],
            sketch_size=self._config.rank,
            dtype=A.dtype,
            device=device,
        )

        # Compute sketch
        Y = A @ Omega

        # Compute core
        Core = Omega.T @ Y

        # Shift for stability
        shift = torch.finfo(Y.dtype).eps * torch.trace(Core)
        Core.diagonal().add_(shift)

        L = torch.linalg.cholesky(Core, upper=False)

        # Get eigendecomposition
        B = torch.linalg.solve_triangular(L, Y.T, upper=False)
        self.U, self.S, _ = torch.linalg.svd(B.T, full_matrices=False)
        self.S = torch.nn.functional.relu(self.S**2 - shift)


def _generate_ortho_embedding(
    dimension: int, sketch_size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Generate an orthogonal random embedding matrix.

    Args:
        dimension (int): Dimension of the original space.
        sketch_size (int): Size of the sketch (number of rows in the embedding).
        dtype (torch.dtype): Data type of the embedding matrix.
        device (torch.device): Device on which to create the embedding matrix.

    Returns:
        torch.Tensor: Orthogonal random embedding matrix of shape
        (sketch_size, dimension).
    """
    # Generate a random Gaussian matrix
    Omega = torch.linalg.qr(
        torch.randn(dimension, sketch_size, dtype=dtype, device=device),
        mode="reduced",
    )[0]

    return Omega
