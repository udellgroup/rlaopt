"""Sketch module for matrix sketching operations.

This module provides an abstract base class for implementing various sketching
techniques used in randomized linear algebra.
"""

from abc import ABC, abstractmethod
from typing import Union

import torch

from .enums import _SketchSide
from rlaopt.linops import LinOpType
from rlaopt.utils import _is_pos_int


class Sketch(ABC):
    """Abstract base class for matrix sketching.

    This class defines the interface for sketching matrices using
    various techniques. Concrete implementations should inherit from
    this class and implement the _generate_embedding method.

    Attributes:
        mode (_SketchSide): An enum indicating the sketching mode
                            (_SketchSide.LEFT or _SketchSide.RIGHT).
        s (int): The sketch size.
        d (int): The dimension of the matrix to be sketched.
        device (torch.device): Device to be used for computations.
        Omega_mat (torch.device): A torch.Tensor representing the sketching matrix.
    """

    def __init__(
        self, mode: str, sketch_size: int, matrix_dim: int, device: torch.device
    ):
        """Initializes the Sketch with given parameters.

        Args:
            mode: A string ('left' or 'right') specifying the sketching mode.
            sketch_size: An integer specifying the size of the sketch.
            matrix_dim: An integer specifying the dimension of the original matrix.
            device: A torch.device object specifying the computation device.

        Raises:
            ValueError: If mode is an invalid value.
        """
        self.mode = _SketchSide._from_str(mode, "mode")
        self.s = sketch_size
        self.d = matrix_dim
        self.device = device

        _is_pos_int(sketch_size, "sketch_size")

        self.Omega_mat = self._generate_embedding()

    @abstractmethod
    def _generate_embedding(self) -> torch.Tensor:
        """Generates the embedding matrix for sketching.

        This method should be implemented by subclasses to generate
        the specific embedding matrix used in the sketching process.

        Returns:
            A torch.Tensor representing the embedding matrix.
        """
        pass

    def _apply_left(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        """Left multiplies input by the sketching matrix: Omega @ x.

        Args:
            x: A torch.Tensor or LinOpType object to be sketched.

        Returns:
            A torch.Tensor representing the result of left sketching.
        """
        return self.Omega_mat @ x

    def _apply_right(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        """Right multiplies input by the sketching matrix: x @ Omega.

        Args:
            x: A torch.Tensor or LinOpType object to be sketched.

        Returns:
            A torch.Tensor representing the result of right sketching.
        """
        return x @ self.Omega_mat

    def _apply_left_trans(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        """Left multiplies input by the transposed sketching matrix: Omega.T @ x.

        Args:
            x: A torch.Tensor or LinOpType object to be sketched.

        Returns:
            A torch.Tensor representing the result of transposed left sketching.
        """
        return self.Omega_mat.T @ x

    def _apply_right_trans(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        """Right multiplies the input by the transposed sketching matrix: x @ Omega.T.

        Args:
            x: A torch.Tensor or LinOpType object to be sketched.

        Returns:
            A torch.Tensor representing the result of transposed right sketching.
        """
        return x @ self.Omega_mat.T
