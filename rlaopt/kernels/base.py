from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Optional

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import LinOp, TwoSidedLinOp, SymmetricLinOp
from rlaopt.utils import _is_torch_tensor

# Global, module-level cache to persist across worker calls
_KERNEL_CACHE: Dict[str, LazyTensor] = {}
_LAZY_TENSOR_CACHE: Dict[str, LazyTensor] = {}


class KernelLinOp(SymmetricLinOp, ABC):
    def __init__(self, A: torch.Tensor, kernel_params: Dict[str, Any]):
        self._check_inputs(A, kernel_params)
        self._A = A
        self._kernel_params = kernel_params
        self._K_lazy = self._get_kernel()
        super().__init__(
            device=A.device,
            shape=torch.Size((A.shape[0], A.shape[0])),
            matvec=lambda x: self.K_lazy @ x,
            matmat=lambda x: self.K_lazy @ x,
        )

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def kernel_params(self) -> Dict[str, Any]:
        return self._kernel_params

    @abstractmethod
    def _check_kernel_params(self, kernel_params: Any):
        pass

    def _check_inputs(self, A: Any, kernel_params: Any):
        _is_torch_tensor(A, "A")
        if A.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got {A.ndim}D tensor.")
        self._check_kernel_params(kernel_params)

    @abstractmethod
    def _kernel_computation(
        self, Ai_lazy: LazyTensor, Aj_lazy: LazyTensor
    ) -> LazyTensor:
        pass

    def _get_kernel(
        self, idx1: Optional[torch.Tensor] = None, idx2: Optional[torch.Tensor] = None
    ):
        if idx1 is None:
            Ai_lazy = LazyTensor(self.A[:, None, :])
        else:
            Ai_lazy = LazyTensor(self.A[idx1][:, None, :])

        if idx2 is None:
            Aj_lazy = LazyTensor(self.A[None, :, :])
        else:
            Aj_lazy = LazyTensor(self.A[idx2][None, :, :])

        K_lazy = self._kernel_computation(Ai_lazy, Aj_lazy)
        return K_lazy

    def _get_kernel_linop(
        self,
        idx1: Optional[torch.Tensor] = None,
        idx2: Optional[torch.Tensor] = None,
        symmetric: bool = False,
    ):
        K = self._get_kernel(idx1, idx2)
        linop_class = SymmetricLinOp if symmetric else LinOp
        return linop_class(
            device=self.device,
            shape=torch.Size(K.shape),
            matvec=lambda x: K @ x,
            matmat=lambda x: K @ x,
        )

    def row_oracle(self, blk: torch.Tensor):
        _is_torch_tensor(blk, "blk")
        return self._get_kernel_linop(idx1=blk, symmetric=False)

    def blk_oracle(self, blk: torch.Tensor):
        _is_torch_tensor(blk, "blk")
        return self._get_kernel_linop(idx1=blk, idx2=blk, symmetric=True)


class _CacheableKernelLinOp(TwoSidedLinOp, ABC):
    """Private implementation of Kernel linear operator with caching."""

    def __init__(
        self,
        A: torch.Tensor,
        kernel_params: dict,
        chunk_idx: torch.Tensor,
        device: torch.device,
    ):
        self._A = A.to(device)
        self._chunk_idx = chunk_idx
        self._kernel_params = kernel_params
        self._unique_id = (
            f"{self._kernel_name()}_{id(self)}_{len(A)}_{kernel_params}_{A.device}"
        )

        super().__init__(
            device=device,
            shape=torch.Size((self._chunk_idx.shape[0], self._A.shape[0])),
            matvec=self._matvec,
            rmatvec=self._rmatvec,
            matmat=self._matvec,
            rmatmat=self._rmatvec,
        )

    @abstractmethod
    def _kernel_name(self) -> str:
        pass

    @abstractmethod
    def _kernel_computation(self) -> LazyTensor:
        pass

    def _get_kernel(self):
        """Get the cached kernel or compute it if not present."""
        global _KERNEL_CACHE

        # Use process ID to ensure cache is per-process
        pid = os.getpid()
        cache_key = f"{pid}_{self._unique_id}"

        if cache_key not in _KERNEL_CACHE:
            print(f"[PID {pid}] Computing kernel for device {self.device}...")

            # Compute kernel and store in the global cache
            _KERNEL_CACHE[cache_key] = self._kernel_computation()

            print(f"[PID {pid}] Kernel cached. Cache size: {len(_KERNEL_CACHE)}")
        else:
            print(f"[PID {pid}] Using cached kernel for device {self.device}")

        return _KERNEL_CACHE[cache_key]

    def _matvec(self, x: torch.Tensor):
        """Matrix-vector product with caching."""
        kernel = self._get_kernel()
        return kernel @ x

    def _rmatvec(self, x: torch.Tensor):
        """Transpose matrix-vector product with caching."""
        kernel = self._get_kernel()
        return kernel.T @ x

    def _clear_cache(self):
        """Clear the kernel cache for this operator."""
        global _KERNEL_CACHE

        pid = os.getpid()
        cache_key = f"{pid}_{self._unique_id}"

        if cache_key in _KERNEL_CACHE:
            del _KERNEL_CACHE[cache_key]
            print(f"[PID {pid}] Cleared kernel cache for device {self.device}")
