from pykeops.torch import LazyTensor
import torch

from rlaopt.kernels.base import (
    KernelLinOp,
    _CacheableKernelLinOp,
    DistributedKernelLinOp,
    _get_cached_lazy_tensor,
)

__all__ = ["RBFLinOp", "DistributedRBFLinOp"]


def _check_kernel_params(kernel_params):
    """Check kernel parameters for RBF kernel."""
    if "sigma" not in kernel_params:
        raise ValueError("Kernel parameters must include 'sigma'.")
    if not isinstance(kernel_params["sigma"], float):
        raise ValueError("Kernel parameter 'sigma' must be a float.")


def _kernel_computation(Ai_lazy, Aj_lazy, kernel_params):
    """Compute RBF kernel matrix."""
    D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2)
    K_lazy = (-D / (2 * kernel_params["sigma"] ** 2)).exp()
    return K_lazy


def _row_oracle_matvec(
    x: torch.Tensor,
    A_mat: torch.Tensor,
    row_idx: torch.Tensor,
    A_chunk: torch.Tensor,
    kernel_params: dict,
) -> torch.Tensor:
    """Compute RBF kernel matrix-vector product for row oracle."""
    # Get cached tensors
    Ab = A_mat[row_idx].to(A_chunk.device)
    Ab_lazy = LazyTensor(Ab[:, None, :])
    A_chunk_lazy = _get_cached_lazy_tensor(A_chunk)

    # Compute kernel and apply
    K_lazy = _kernel_computation(Ab_lazy, A_chunk_lazy, kernel_params)
    return K_lazy @ x


class RBFLinOp(KernelLinOp):
    def __init__(self, A, kernel_params):
        super().__init__(A=A, kernel_params=kernel_params)

    def _check_kernel_params(self, kernel_params):
        _check_kernel_params(kernel_params)

    def _kernel_computation(self, Ai_lazy, Aj_lazy):
        return _kernel_computation(
            Ai_lazy=Ai_lazy, Aj_lazy=Aj_lazy, kernel_params=self.kernel_params
        )


class _CacheableRBFLinOp(_CacheableKernelLinOp):
    def __init__(self, A, kernel_params, chunk_idx, device):
        super().__init__(
            A=A, kernel_params=kernel_params, chunk_idx=chunk_idx, device=device
        )

    def _kernel_name(self):
        return "rbf_kernel"

    def _kernel_computation(self, Ab_lazy, A_lazy):
        return _kernel_computation(
            Ai_lazy=Ab_lazy, Aj_lazy=A_lazy, kernel_params=self.kernel_params
        )


class DistributedRBFLinOp(DistributedKernelLinOp):
    """Distributed RBF linear operator with row and block oracles that share worker
    processes."""

    def __init__(
        self,
        A,
        kernel_params,
        devices,
        compute_device=None,
    ):
        """Initialize the distributed RBF linear operator."""
        super().__init__(
            cacheable_kernel_class=_CacheableRBFLinOp,
            A=A,
            kernel_params=kernel_params,
            devices=devices,
            compute_device=compute_device,
        )

    def _check_kernel_params(self, kernel_params):
        _check_kernel_params(kernel_params)

    def _get_row_oracle_matvec_fn(self):
        return _row_oracle_matvec

    def _blk_oracle_matvec(self, x, Abi_lazy, Abj_lazy):
        """Compute RBF kernel matrix-vector product for block oracle."""
        K_lazy = _kernel_computation(
            Ai_lazy=Abi_lazy, Aj_lazy=Abj_lazy, kernel_params=self.kernel_params
        )
        return K_lazy @ x
