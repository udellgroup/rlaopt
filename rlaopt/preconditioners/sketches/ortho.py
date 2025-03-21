"""Orthonormal sketch implementation for matrix sketching.

This module provides a concrete implementation of the Sketch class
using orthonormal matrices for sketching.

Typical usage example:

  ortho_sketch = Ortho("left", 100, 1000, device)
  sketched_matrix = ortho_sketch.apply(matrix)
"""

import torch
from rlaopt.preconditioners.sketches.sketch import Sketch


class Ortho(Sketch):
    """Orthonormal sketch class for matrix sketching.

    This class implements the Sketch abstract base class using
    orthonormal matrices for the sketching process.

    Attributes:
        Inherited from Sketch class.
    """

    def __init__(
        self, mode: str, sketch_size: int, matrix_dim: int, device: torch.device
    ):
        """Initializes the Orthonormal sketch with given parameters.

        Args:
            mode: A string specifying the sketching mode ('left' or 'right').
            sketch_size: An integer specifying the size of the sketch.
            matrix_dim: An integer specifying the dimension of the original matrix.
            device: A torch.device object specifying the computation device.
        """
        super().__init__(mode, sketch_size, matrix_dim, device)

    def _generate_embedding(self) -> torch.Tensor:
        """Generates the orthonormal embedding matrix for sketching.

        This method creates an orthonormal matrix by first generating
        a random Gaussian matrix and then applying QR decomposition
        to obtain the orthonormal factor. The resulting matrix is
        transposed if the mode is set to "left".

        Returns:
            A torch.Tensor representing the orthonormal embedding matrix.

        Note:
            The orthonormal matrix is generated using the QR decomposition
            of a random Gaussian matrix. This ensures that the columns
            (or rows, depending on the mode) of the embedding matrix
            are orthonormal, which can provide certain guarantees for
            the sketching process.
        """
        Omega_mat = torch.linalg.qr(
            torch.randn(self.d, self.s, device=self.device), mode="reduced"
        )[0]
        if self.mode == "left":
            Omega_mat = Omega_mat.T
        return Omega_mat
