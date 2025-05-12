"""Rademacher sketch implementation for matrix sketching.

This module provides a concrete implementation of the Sketch class
using Rademacher random matrices for sketching.

Typical usage example:

  Omega = Rademacher("left", 100, 1000, device)
  sketched_matrix = Omega.apply_left(matrix)
"""
import torch

from .enums import _SketchSide
from .sketch import Sketch


class Rademacher(Sketch):
    """Rademacher sketch class for matrix sketching.

    This class implements the Sketch abstract base class using
    random sign Rademacher matrices for the sketching process.

    Attributes:
        Inherited from Sketch class.
    """

    def __init__(self, mode, sketch_size, matrix_dim, dtype, device):
        """Initializes the Rademacher sketch with given parameters."""
        super().__init__(mode, sketch_size, matrix_dim, dtype, device)

    def _generate_embedding(self) -> torch.Tensor:
        """Generates the Rademacher embedding matrix for sketching.

        This method creates a matrix with entries +1 or -1, each with probability 1/2.
        The matrix is transposed if the mode is set to _SketchSide.RIGHT.

        Returns:
            A torch.Tensor representing the Rademacher embedding matrix.
        """
        Omega_mat = (
            2
            * torch.randint(
                0, 2, (self.s, self.d), dtype=self.dtype, device=self.device
            )
            - 1
        )
        if self.mode == _SketchSide.RIGHT:
            Omega_mat = Omega_mat.T
        return Omega_mat.contiguous()
