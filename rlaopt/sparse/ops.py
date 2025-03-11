import torch
from torch import Tensor

__all__ = ["_get_row_slice", "_csc_matvec"]


def _get_row_slice(sparse_tensor: Tensor, row_indices: Tensor) -> Tensor:
    """Get a slice of rows from a sparse tensor.

    Args:
        sparse_tensor (Tensor): The sparse CSR tensor to slice.
        row_indices (Tensor): The row indices to slice.

    Returns:
        Tensor: The slice of rows from the sparse tensor.
    """
    return torch.ops.rlaopt.get_row_slice.default(sparse_tensor, row_indices)


def _csc_matvec(sparse_tensor: Tensor, dense_vector: Tensor) -> Tensor:
    """Multiply a sparse CSC tensor with a dense vector.

    Args:
        sparse_tensor (Tensor): The sparse CSC tensor.
        dense_vector (Tensor): The dense vector.

    Returns:
        Tensor: The result of the multiplication.
    """
    return torch.ops.rlaopt.csc_matvec.default(sparse_tensor, dense_vector)
