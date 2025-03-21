from functools import partial
import os
from typing import Dict, Set

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    SymmetricLinOp,
    DistributionMode,
    DistributedSymmetricLinOp,
)

__all__ = ["RBFLinOp", "DistributedRBFLinOpV1"]


# Global, module-level cache to persist across worker calls
_PROCESS_KERNEL_CACHE: Dict[str, LazyTensor] = {}


def _matvec(x: torch.Tensor, K: LazyTensor):
    return K @ x


class RBFLinOp(SymmetricLinOp):
    def __init__(self, A: torch.Tensor, sigma: float):
        """Initialize the RBF kernel.

        Args:
            A (torch.Tensor): The input data.
            sigma (float): The bandwidth parameter for the RBF kernel.
        """
        self.A = A
        self.sigma = sigma
        K = self._get_K()
        super().__init__(
            device=A.device,
            shape=torch.Size((A.shape[0], A.shape[0])),
            matvec=partial(_matvec, K=K),
            matmat=partial(_matvec, K=K),
        )

    def _get_K(
        self,
        idx1: torch.Tensor = None,
        idx2: torch.Tensor = None,
    ):
        if idx1 is None:
            Ai_lazy = LazyTensor(self.A[:, None, :])
        else:
            Ai_lazy = LazyTensor(self.A[idx1][:, None, :])

        if idx2 is None:
            Aj_lazy = LazyTensor(self.A[None, :, :])
        else:
            Aj_lazy = LazyTensor(self.A[idx2][None, :, :])

        D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2)
        K = (-D / (2 * self.sigma**2)).exp()
        return K

    def _get_K_linop(
        self,
        idx1: torch.Tensor = None,
        idx2: torch.Tensor = None,
        symmetric: bool = False,
    ):
        K = self._get_K(idx1=idx1, idx2=idx2)
        linop_class = SymmetricLinOp if symmetric else LinOp
        K_linop = linop_class(
            device=self.A.device,
            shape=torch.Size(K.shape),
            matvec=lambda x: K @ x,
            matmat=lambda x: K @ x,
        )
        return K_linop

    def row_oracle(self, blk: torch.Tensor):
        return self._get_K_linop(idx1=blk, symmetric=False)

    def blk_oracle(self, blk: torch.Tensor):
        return self._get_K_linop(idx1=blk, idx2=blk, symmetric=True)


class _CacheableRBFLinOp(TwoSidedLinOp):
    """Private implementation of RBF linear operator with caching."""

    def __init__(
        self,
        A: torch.Tensor,
        chunk_idx: torch.Tensor,
        sigma: float,
        device: torch.device,
    ):
        """Initialize the RBF kernel operator."""
        # Store the parameters needed to compute kernels
        self.A = A.to(device)
        self.chunk_idx = chunk_idx
        self.sigma = sigma

        # Generate a unique ID for this operator instance
        # This ID will be used to access the global cache
        self._unique_id = f"rbf_kernel_{id(self)}_{len(chunk_idx)}_{sigma}_{device}"

        # Initialize the operator
        super().__init__(
            device=device,
            shape=torch.Size((chunk_idx.shape[0], A.shape[0])),
            matvec=self._matvec,
            rmatvec=self._rmatvec,
            matmat=self._matvec,
            rmatmat=self._rmatvec,
        )

    def _get_kernel(self):
        """Get the cached kernel or compute it if not present."""
        global _PROCESS_KERNEL_CACHE

        # Use process ID to ensure cache is per-process
        pid = os.getpid()
        cache_key = f"{pid}_{self._unique_id}"

        if cache_key not in _PROCESS_KERNEL_CACHE:
            print(f"[PID {pid}] Computing kernel for device {self.device}...")

            # Compute the kernel
            Ab_lazy = LazyTensor(self.A[self.chunk_idx][:, None, :])
            A_lazy = LazyTensor(self.A[None, :, :])
            D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
            kernel = (-D / (2 * self.sigma**2)).exp()

            # Store in the global cache
            _PROCESS_KERNEL_CACHE[cache_key] = kernel
            print(
                f"[PID {pid}] Kernel cached. Cache size: {len(_PROCESS_KERNEL_CACHE)}"
            )
        else:
            print(f"[PID {pid}] Using cached kernel for device {self.device}")

        return _PROCESS_KERNEL_CACHE[cache_key]

    def _matvec(self, x: torch.Tensor):
        """Forward matrix-vector product with caching."""
        kernel = self._get_kernel()
        return kernel @ x

    def _rmatvec(self, x: torch.Tensor):
        """Transpose matrix-vector product with caching."""
        kernel = self._get_kernel()
        return kernel.T @ x

    def clear_cache(self):
        """Clear the kernel cache for this operator."""
        global _PROCESS_KERNEL_CACHE

        pid = os.getpid()
        cache_key = f"{pid}_{self._unique_id}"

        if cache_key in _PROCESS_KERNEL_CACHE:
            del _PROCESS_KERNEL_CACHE[cache_key]
            print(f"[PID {pid}] Cleared kernel cache for device {self.device}")


class DistributedRBFLinOpV1(DistributedSymmetricLinOp):
    """Distributed RBF linear operator with per-worker caching."""

    def __init__(self, A: torch.Tensor, sigma: float, devices: Set[torch.device]):
        # Clean the global cache at initialization
        global _PROCESS_KERNEL_CACHE
        _PROCESS_KERNEL_CACHE.clear()
        print(f"Initialized with clean cache. PID: {os.getpid()}")

        A_chunk_idx = torch.chunk(torch.arange(A.shape[0]), len(devices), dim=0)

        # Create _CacheableRBFLinOp instances for each chunk
        rbf_ops = []
        for device, chunk_idx in zip(devices, A_chunk_idx):
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

    def shutdown(self):
        """Extend shutdown to clear caches."""
        # Clear kernel caches to free memory
        for op in self.rbf_ops:
            op.clear_cache()

        # Also clear the global cache
        global _PROCESS_KERNEL_CACHE
        _PROCESS_KERNEL_CACHE.clear()
        print(f"Cleared global cache on shutdown. PID: {os.getpid()}")

        # Call parent shutdown
        super().shutdown()
