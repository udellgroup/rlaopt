from typing import Any, Callable, Dict

import torch
from pykeops.torch import LazyTensor

from .base import _get_cached_lazy_tensor


def _check_kernel_params(kernel_params: Dict[str, Any]):
    """Check kernel parameters."""
    if "lengthscale" not in kernel_params:
        raise ValueError("Kernel parameters must include 'lengthscale'.")


def _row_oracle_matvec(
    x: torch.Tensor,
    A1: torch.Tensor,
    A2_chunk: torch.Tensor,
    blk: torch.Tensor,
    kernel_params: Dict[str, Any],
    kernel_computation: Callable,
) -> torch.Tensor:
    """Compute kernel matrix-vector product for row oracle.

    Necessary for distributed kernel linear operator due to multiprocessing issues.
    """
    # Get cached tensors
    A1b = A1[blk].to(A2_chunk.device)
    A1b_lazy = LazyTensor(A1b[:, None, :])
    A2_chunk_lazy = _get_cached_lazy_tensor(A2_chunk)

    # Compute kernel and apply
    K_lazy = kernel_computation(A1b_lazy, A2_chunk_lazy, kernel_params)
    return K_lazy @ x


def _block_chunk_matvec(
    x: torch.Tensor,
    device: torch.device,
    A1: torch.Tensor,
    A2: torch.Tensor,
    blk_chunk: torch.Tensor,
    blk: torch.Tensor,
    kernel_params: Dict[str, Any],
    kernel_computation: Callable,
) -> torch.Tensor:
    """Compute the matrix-vector product for a chunk of the block kernel matrix.

    Necessary for distributed kernel linear operator due to multiprocessing issues.
    """
    # Get the data for this chunk
    A1_blk_chunk = A1[blk_chunk].to(device)
    A2_blk_full = A2[blk].to(device)

    # Create LazyTensors
    A1b_lazy = LazyTensor(A1_blk_chunk[:, None, :])
    A2b_lazy = LazyTensor(A2_blk_full[None, :, :])

    # Compute kernel and matrix-vector product
    K_lazy = kernel_computation(A1b_lazy, A2b_lazy, kernel_params)
    return K_lazy @ x
