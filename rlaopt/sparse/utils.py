from typing import Union
import torch


def _convert_indices_to_tensor(
    indices: Union[torch.Tensor, slice, int, list[int]], num_rows: int
) -> torch.Tensor:
    """Convert various index types to a PyTorch tensor of indices.

    Args:
        indices: Input indices in one of these formats:
            - torch.Tensor: Tensor of indices
            - slice: Python slice object (e.g., [1:10:2])
            - int: Single index
            - list: List of indices
        num_rows: Total number of rows in the original tensor

    Returns:
        torch.Tensor: A 1D long tensor containing the indices

    Raises:
        TypeError: If indices format is invalid
        IndexError: If indices are out of bounds
    """
    # Handle different types of indices
    if isinstance(indices, slice):
        # Use the slice.indices() function to get (start, stop, step)
        start, stop, step = indices.indices(num_rows)

        # Create tensor of indices from the slice parameters
        tensor_indices = torch.arange(start, stop, step)

    elif isinstance(indices, int):
        # Convert single integer to tensor with one element
        if indices < 0:
            indices = num_rows + indices
        tensor_indices = torch.tensor([indices])

    elif isinstance(indices, list):
        # Convert list to tensor
        tensor_indices = torch.tensor(indices)

    elif isinstance(indices, torch.Tensor):
        # Already a tensor, just make a copy to ensure it's not modified
        tensor_indices = indices.clone()

    else:
        # Handle unsupported types
        raise TypeError(
            f"Slicing indices must be a tensor, slice, int, or list. "
            f"Got type {type(indices).__name__}"
        )

    # Ensure indices are long type for indexing
    tensor_indices = tensor_indices.long()

    # Check for out of bounds indices
    if tensor_indices.numel() > 0:
        if torch.min(tensor_indices) < 0 or torch.max(tensor_indices) >= num_rows:
            raise IndexError(
                f"Slicing indices out of bounds. Valid range: [0, {num_rows-1}]"
            )

    return tensor_indices
