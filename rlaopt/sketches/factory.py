"""This module provides a factory for creating various types of sketches.

It contains a helper function to instantiate different sketch objects based on given
parameters and configurations.
"""

import torch

from .sketch import Sketch
from .gauss import Gauss
from .sparse import Sparse
from .ortho import Ortho

SKETCHES = {"gauss": Gauss, "ortho": Ortho, "sparse": Sparse}


__all__ = ["get_sketch"]


def get_sketch(
    name: str, mode: str, sketch_size: int, matrix_dim: int, device: torch.device
) -> Sketch:
    """Factory function to create a Sketch object.

    This function creates and returns an instance of a specific Sketch subclass
    based on the provided name.

    Args:
        name (str): The name of the sketching technique to use.
                    Valid options are "gauss", "ortho", and "sparse".
        mode (str): The sketching mode. Can be specified as "left" or "right".
        sketch_size (int): The target dimension of the sketch.
        matrix_dim (int): The dimension of the original matrix.
        device (torch.device): The device on which to perform computations.

    Returns:
        Sketch: An instance of the specified Sketch subclass.

    Raises:
        ValueError: If the provided name is not a valid sketching technique.

    Example:
        >>> sketch = get_sketch("gauss", "left", 100, 1000, torch.device("cuda"))
    """
    if name not in SKETCHES:
        valid_sketches = ", ".join(SKETCHES.keys())
        raise ValueError(
            f"{name} is not a valid sketching matrix. "
            f"Valid sketching matrices are: {valid_sketches}"
        )
    sketch_class = SKETCHES[name]
    return sketch_class(mode, sketch_size, matrix_dim, device)
