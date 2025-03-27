"""Gaussian sketch implementation for matrix sketching.

This module provides a concrete implementation of the Sketch class
using Gaussian random matrices for sketching.

Typical usage example:

  Omega = Gauss("left", 100, 1000, device)
  sketched_matrix = gauss_sketch.apply_left(matrix)
"""

import torch

from .sketch import Sketch


class Gauss(Sketch):
    """Gaussian sketch class for matrix sketching.

    This class implements the Sketch abstract base class using
    Gaussian random matrices for the sketching process.

    Attributes:
        Inherited from Sketch class.
    """

    def __init__(
        self, mode: str, sketch_size: int, matrix_dim: int, device: torch.device
    ):
        """Initializes the Gaussian sketch with given parameters.

        Args:
            mode: A string specifying the sketching mode ('left' or 'right').
            sketch_size: An integer specifying the size of the sketch.
            matrix_dim: An integer specifying the dimension of the original matrix.
            device: A torch.device object specifying the computation device.
        """
        super().__init__(mode, sketch_size, matrix_dim, device)

    def _generate_embedding(self) -> torch.Tensor:
        """Generates the Gaussian embedding matrix for sketching.

        This method creates a Gaussian random matrix and normalizes it
        according to the sketch size. The matrix is transposed if the
        mode is set to "right".

        Returns:
            A torch.Tensor representing the Gaussian embedding matrix.

        Note:
            Normalizing by 1 / np.sqrt(sketch_size) ensures
            Omega.T @ Omega is an expected isometry.
        """
        Omega_mat = torch.randn(self.s, self.d, device=self.device) / (self.s) ** (0.5)

        if self.mode == "right":
            Omega_mat = Omega_mat.T

        return Omega_mat
