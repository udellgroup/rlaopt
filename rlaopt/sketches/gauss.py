"""Gaussian sketch implementation for matrix sketching.

This module provides a concrete implementation of the Sketch class
using Gaussian random matrices for sketching.

Typical usage example:

  Omega = Gauss("left", 100, 1000, device)
  sketched_matrix = gauss_sketch.apply_left(matrix)
"""

import torch

from .enums import _SketchSide
from .sketch import Sketch


class Gauss(Sketch):
    """Gaussian sketch class for matrix sketching.

    This class implements the Sketch abstract base class using
    Gaussian random matrices for the sketching process.

    Attributes:
        Inherited from Sketch class.
    """

    def __init__(self, mode, sketch_size, matrix_dim, dtype, device):
        """Initializes the Gaussian sketch with given parameters."""
        super().__init__(mode, sketch_size, matrix_dim, dtype, device)

    def _generate_embedding(self) -> torch.Tensor:
        """Generates the Gaussian embedding matrix for sketching.

        This method creates a Gaussian random matrix and normalizes it
        according to the sketch size. The matrix is transposed if the
        mode is set to _SketchSide.RIGHT.

        Returns:
            A torch.Tensor representing the Gaussian embedding matrix.

        Note:
            Normalizing by 1 / np.sqrt(sketch_size) ensures
            Omega.T @ Omega is an expected isometry.
        """
        Omega_mat = torch.randn(
            self.s, self.d, dtype=self.dtype, device=self.device
        ) / (self.s) ** (0.5)

        if self.mode == _SketchSide.RIGHT:
            Omega_mat = Omega_mat.T
        return Omega_mat.contiguous()  # contiguous is to avoid KeOps contiguity warning
