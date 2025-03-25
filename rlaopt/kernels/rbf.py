from functools import partial

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import LinOp, DistributionMode
from rlaopt.linops.distributed import _DistributedLinOp
from rlaopt.kernels.base import (
    KernelLinOp,
    _CacheableKernelLinOp,
    DistributedKernelLinOp,
    _get_cached_lazy_tensor,
)

__all__ = ["RBFLinOp", "DistributedRBFLinOp"]


class RBFLinOp(KernelLinOp):
    def __init__(self, A, kernel_params):
        super().__init__(A=A, kernel_params=kernel_params)

    def _check_kernel_params(self, kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Kernel parameters must include 'sigma'.")
        if not isinstance(kernel_params["sigma"], float):
            raise ValueError("Kernel parameter 'sigma' must be a float.")

    def _kernel_computation(self, Ai_lazy, Aj_lazy):
        D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2)
        K_lazy = (-D / (2 * self.kernel_params["sigma"] ** 2)).exp()
        return K_lazy


class _CacheableRBFLinOp(_CacheableKernelLinOp):
    def __init__(self, A, kernel_params, chunk_idx, device):
        super().__init__(
            A=A, kernel_params=kernel_params, chunk_idx=chunk_idx, device=device
        )

    def _kernel_name(self):
        return "rbf_kernel"

    def _kernel_computation(self):
        Ab_lazy = LazyTensor(self.A[self.chunk_idx][:, None, :])
        A_lazy = LazyTensor(self.A[None, :, :])
        D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
        kernel = (-D / (2 * self.kernel_params["sigma"] ** 2)).exp()
        return kernel


def _rbf_row_oracle_matvec(
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
    D = ((Ab_lazy - A_chunk_lazy) ** 2).sum(dim=2)
    K = (-D / (2 * kernel_params["sigma"] ** 2)).exp()
    return K @ x


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
            A=A,
            kernel_params=kernel_params,
            devices=devices,
            compute_device=compute_device,
        )

    def _check_kernel_params(self, kernel_params):
        if "sigma" not in kernel_params:
            raise ValueError("Kernel parameters must include 'sigma'.")
        if not isinstance(kernel_params["sigma"], float):
            raise ValueError("Kernel parameter 'sigma' must be a float.")

    def _create_kernel_operators(self):
        """Create RBF kernel operators for each chunk."""
        ops = []
        for device, chunk_idx in zip(self.devices, self.A_row_chunks):
            ops.append(
                _CacheableRBFLinOp(
                    A=self.A_mat,
                    kernel_params=self.kernel_params,
                    chunk_idx=chunk_idx,
                    device=device,
                )
            )
        return ops

    def row_oracle(self, blk):
        # Create operators for each device
        row_ops = []
        for i, device in enumerate(self.devices):
            A_chunk = self.A_chunks[i]
            kernel_params = self.kernel_params
            A_mat = self.A_mat  # Capture these variables explicitly

            matvec_fn = partial(
                _rbf_row_oracle_matvec,
                A_mat=A_mat,
                row_idx=blk,
                A_chunk=A_chunk,
                kernel_params=kernel_params,
            )

            # Create a LinOp with the matvec function
            row_ops.append(
                LinOp(
                    device=device,
                    shape=torch.Size((blk.shape[0], A_chunk.shape[0])),
                    matvec=matvec_fn,
                    matmat=matvec_fn,
                )
            )

        # Create a distributed operator that reuses our workers
        return _DistributedLinOp(
            shape=torch.Size((blk.shape[0], self.A_mat.shape[0])),
            A=row_ops,
            distribution_mode=DistributionMode.COLUMN,
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
            is_new=False,  # Important: reuse existing workers
        )

    def _blk_oracle_matvec(
        self, x: torch.Tensor, blk_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute RBF kernel matrix-vector product for block oracle."""
        A_blk = self.A_mat[blk_idx].to(self.compute_device)

        # Compute kernel and apply
        Ab_lazy = LazyTensor(A_blk[:, None, :])
        A_lazy = LazyTensor(A_blk[None, :, :])
        D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
        K = (-D / (2 * self.kernel_params["sigma"] ** 2)).exp()
        return K @ x
