from typing import Any, Callable, Dict

import torch
from pykeops.torch import LazyTensor

from .base import _get_cached_lazy_tensor
from rlaopt.utils import _is_pos_float  # noqa: F401


def _check_kernel_params(kernel_params: Dict[str, Any]):
    """Check kernel parameters."""
    if "lengthscale" not in kernel_params:
        raise ValueError("Kernel parameters must include 'lengthscale'.")
    # _is_pos_float(kernel_params["sigma"], 'kernel_params["sigma"]')


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
