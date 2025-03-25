from functools import partial
import os
from typing import Dict, Set, Optional

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import (
    LinOp,
    SymmetricLinOp,
    DistributionMode,
    DistributedLinOp,
    DistributedSymmetricLinOp,
)
from rlaopt.linops.distributed import _DistributedLinOp
from rlaopt.kernels.base import KernelLinOp, _CacheableKernelLinOp

__all__ = ["RBFLinOp", "DistributedRBFLinOp"]


# Global, module-level cache to persist across worker calls
_KERNEL_CACHE: Dict[str, LazyTensor] = {}
_LAZY_TENSOR_CACHE: Dict[str, LazyTensor] = {}


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


def _get_cached_lazy_tensor(A):
    """Get a cached LazyTensor or create a new one."""
    global _LAZY_TENSOR_CACHE

    pid = os.getpid()
    cache_key = f"{pid}_lazy_{id(A)}_{A.device}"

    if cache_key not in _LAZY_TENSOR_CACHE:
        _LAZY_TENSOR_CACHE[cache_key] = LazyTensor(A[None, :, :])

    return _LAZY_TENSOR_CACHE[cache_key]


def _row_oracle_matvec(x, A, row_idx, A_chunk, sigma):
    """Matrix-vector product for row oracle, with LazyTensor caching."""
    # Get cached tensors
    Ab = A[row_idx].to(A_chunk.device)
    Ab_lazy = LazyTensor(Ab[:, None, :])
    A_chunk_lazy = _get_cached_lazy_tensor(A_chunk)

    # Compute kernel and apply
    D = ((Ab_lazy - A_chunk_lazy) ** 2).sum(dim=2)
    K = (-D / (2 * sigma**2)).exp()
    return K @ x


class DistributedRBFLinOp(DistributedSymmetricLinOp):
    """Distributed RBF linear operator with row and block oracles that share worker
    processes."""

    def __init__(
        self,
        A: torch.Tensor,
        sigma: float,
        devices: Set[torch.device],
        compute_device: Optional[torch.device] = None,
    ):
        """Initialize the distributed RBF linear operator.

        Args:
            A: Input data tensor
            sigma: RBF kernel bandwidth parameter
            devices: Set of devices to distribute computation across
            compute_device: Device to use for block
            computation (default: first device in devices)
        """
        # Clean the global caches at initialization
        global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
        _KERNEL_CACHE.clear()
        _LAZY_TENSOR_CACHE.clear()
        print(f"Initialized with clean caches. PID: {os.getpid()}")

        # Save parameters
        self.A = A  # Keep original tensor for oracles
        self.sigma = sigma
        self.devices = list(devices)
        self.compute_device = compute_device or self.devices[0]

        # Create row partitioning
        self.A_row_chunks = torch.chunk(torch.arange(A.shape[0]), len(devices), dim=0)

        # Create chunks of data for each device
        self.A_chunks = []
        for device, chunk_idx in zip(self.devices, self.A_row_chunks):
            # We keep chunks on each device for row_oracle
            self.A_chunks.append(A[chunk_idx].to(device))

        # Create _CacheableRBFLinOp instances for each chunk
        rbf_ops = []
        for device, chunk_idx in zip(self.devices, self.A_row_chunks):
            rbf_ops.append(
                _CacheableRBFLinOp(A=A, chunk_idx=chunk_idx, sigma=sigma, device=device)
            )

        # Initialize the distributed operator
        super().__init__(
            shape=torch.Size((A.shape[0], A.shape[0])),
            A=rbf_ops,
            distribution_mode=DistributionMode.ROW,
        )

        # Store references for cleanup
        self.rbf_ops = rbf_ops

    def row_oracle(self, blk: torch.Tensor) -> DistributedLinOp:
        """Get a distributed operator for specific rows.

        This uses LazyTensor caching instead of kernel caching.

        Args:
            blk: Indices of rows to extract

        Returns:
            A distributed linear operator for the specified rows
        """
        # Create operators for each device with the efficient matvec function
        row_ops = []
        for i, device in enumerate(self.devices):
            chunk = self.A_chunks[i]

            matvec = partial(
                _row_oracle_matvec,
                A=self.A,
                row_idx=blk,
                A_chunk=chunk,
                sigma=self.sigma,
            )

            # Create a LinOp with the cached matvec function
            row_ops.append(
                LinOp(
                    device=device,
                    shape=torch.Size((blk.shape[0], chunk.shape[0])),
                    matvec=matvec,
                    matmat=matvec,
                )
            )

        # Create a distributed operator that reuses our workers
        return _DistributedLinOp(
            shape=torch.Size((blk.shape[0], self.A.shape[0])),
            A=row_ops,
            distribution_mode=DistributionMode.COLUMN,
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
            is_new=False,  # Important: reuse existing workers
        )

    def blk_oracle(self, blk: torch.Tensor) -> SymmetricLinOp:
        """Get a simple, non-cached symmetric operator for a block.

        Args:
            blk: Indices defining the block

        Returns:
            A symmetric linear operator for the specified block
        """
        # Use device-specific data
        A_blk = self.A[blk].to(self.compute_device)

        # Simple implementation without caching
        def _blk_matvec(x: torch.Tensor) -> torch.Tensor:
            Ab_lazy = LazyTensor(A_blk[:, None, :])
            A_lazy = LazyTensor(A_blk[None, :, :])
            D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
            K = (-D / (2 * self.sigma**2)).exp()
            return K @ x

        return SymmetricLinOp(
            device=self.compute_device,
            shape=torch.Size((blk.shape[0], blk.shape[0])),
            matvec=_blk_matvec,
            matmat=_blk_matvec,
        )

    def shutdown(self):
        """Extend shutdown to clear caches."""
        # Clear kernel caches to free memory
        for op in self.rbf_ops:
            op._clear_cache()

        # Clear the global caches
        global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
        _KERNEL_CACHE.clear()
        _LAZY_TENSOR_CACHE.clear()
        print(f"Cleared global caches on shutdown. PID: {os.getpid()}")

        # Call parent shutdown
        super().shutdown()
