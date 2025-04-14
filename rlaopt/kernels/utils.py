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
    A_mat: torch.Tensor,
    row_idx: torch.Tensor,
    A_chunk: torch.Tensor,
    kernel_params: Dict[str, Any],
    kernel_computation: Callable,
) -> torch.Tensor:
    """Compute kernel matrix-vector product for row oracle.

    Necessary for distributed kernel linear operator due to multiprocessing issues.
    """
    # Get cached tensors
    Ab = A_mat[row_idx].to(A_chunk.device)
    Ab_lazy = LazyTensor(Ab[:, None, :])
    A_chunk_lazy = _get_cached_lazy_tensor(A_chunk)

    # Compute kernel and apply
    K_lazy = kernel_computation(Ab_lazy, A_chunk_lazy, kernel_params)
    return K_lazy @ x


def _block_chunk_matvec(
    x: torch.Tensor,
    device: torch.device,
    A_mat: torch.Tensor,
    blk_chunk: torch.Tensor,
    blk: torch.Tensor,
    kernel_params: Dict[str, Any],
    kernel_computation: Callable,
) -> torch.Tensor:
    """Compute the matrix-vector product for a chunk of the block kernel matrix.

    Necessary for distributed kernel linear operator due to multiprocessing issues.
    """
    # Get the data for this chunk
    A_blk_chunk = A_mat[blk_chunk].to(device)
    A_blk_full = A_mat[blk].to(device)

    # Create LazyTensors
    Abi_lazy = LazyTensor(A_blk_chunk[:, None, :])
    Abj_lazy = LazyTensor(A_blk_full[None, :, :])

    # Compute kernel and matrix-vector product
    K_lazy = kernel_computation(Abi_lazy, Abj_lazy, kernel_params)
    return K_lazy @ x
