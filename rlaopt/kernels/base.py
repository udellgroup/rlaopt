from abc import ABC, abstractmethod
from functools import partial
import os
from typing import Any, Dict, Optional, List, Set

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    SymmetricLinOp,
    DistributionMode,
    DistributedLinOp,
    DistributedSymmetricLinOp,
)
from rlaopt.linops.distributed import _DistributedLinOp
from rlaopt.utils import _is_torch_tensor, _is_torch_device, _is_dict, _is_set

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
        _is_dict(kernel_params, "kernel_params")
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
        kernel_params: Dict[str, Any],
        chunk_idx: torch.Tensor,
        device: torch.device,
    ):
        self._A = A.to(device)
        self._kernel_params = kernel_params
        self._chunk_idx = chunk_idx
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

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def kernel_params(self) -> Dict[str, Any]:
        return self._kernel_params

    @property
    def chunk_idx(self) -> torch.Tensor:
        return self._chunk_idx

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


def _get_cached_lazy_tensor(A: torch.Tensor) -> LazyTensor:
    """Get a cached LazyTensor or create a new one."""
    global _LAZY_TENSOR_CACHE

    pid = os.getpid()
    cache_key = f"{pid}_lazy_{id(A)}_{A.device}"

    if cache_key not in _LAZY_TENSOR_CACHE:
        _LAZY_TENSOR_CACHE[cache_key] = LazyTensor(A[None, :, :])

    return _LAZY_TENSOR_CACHE[cache_key]


class DistributedKernelLinOp(DistributedSymmetricLinOp, ABC):
    """Abstract base class for distributed kernel linear operators."""

    def __init__(
        self,
        A: torch.Tensor,
        kernel_params: Dict[str, Any],
        devices: Set[torch.device],
        compute_device: Optional[torch.device] = None,
    ):
        """Initialize the distributed kernel linear operator.

        Args:
            A: Input data tensor
            kernel_params: Dictionary of kernel parameters
            devices: Set of devices to distribute computation across
            compute_device: Device to use for block computation
            (default: first device in devices)
        """
        # Clean the global caches at initialization
        global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
        _KERNEL_CACHE.clear()
        _LAZY_TENSOR_CACHE.clear()
        print(f"Initialized with clean caches. PID: {os.getpid()}")

        # Save parameters
        self._check_inputs(A, kernel_params, devices, compute_device)
        self._A_mat = A  # Keep original tensor for oracles
        self._kernel_params = kernel_params
        self.devices = list(devices)
        self.compute_device = compute_device or self.devices[0]

        # Create row partitioning
        self.A_row_chunks = torch.chunk(
            torch.arange(self._A_mat.shape[0]), len(self.devices), dim=0
        )

        # Create chunks of data for each device
        self.A_chunks = []
        for device, chunk_idx in zip(self.devices, self.A_row_chunks):
            # We keep chunks on each device for row_oracle
            self.A_chunks.append(self._A_mat[chunk_idx].to(device))

        # Create cacheable kernel operators for each chunk
        kernel_ops = self._create_kernel_operators()

        # Initialize the distributed operator
        super().__init__(
            shape=torch.Size((self._A_mat.shape[0], self._A_mat.shape[0])),
            A=kernel_ops,
            distribution_mode=DistributionMode.ROW,
        )

        # Store references for cleanup
        self.kernel_ops = kernel_ops

    @property
    def A_mat(self) -> torch.Tensor:
        return self._A_mat

    @property
    def kernel_params(self) -> Dict[str, Any]:
        return self._kernel_params

    @abstractmethod
    def _check_kernel_params(self, kernel_params: Any):
        pass

    def _check_inputs(
        self,
        A: Any,
        kernel_params: Any,
        devices: Any,
        compute_device: Any,
    ):
        _is_torch_tensor(A, "A")
        if A.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got {A.ndim}D tensor.")
        _is_dict(kernel_params, "kernel_params")
        self._check_kernel_params(kernel_params)
        _is_set(devices, "devices")
        if len(devices) == 0:
            raise ValueError("devices must be a non-empty set.")
        if not all(isinstance(d, torch.device) for d in devices):
            raise ValueError("All elements in devices must be torch.device instances.")
        if compute_device is not None:
            _is_torch_device(compute_device, "compute_device")
            if compute_device not in devices:
                raise ValueError("compute_device must be in the set of devices.")

    @abstractmethod
    def _create_kernel_operators(self) -> List[_CacheableKernelLinOp]:
        """Create the kernel operators for each chunk.

        Returns:
            List of kernel operators, one for each device/chunk
        """
        pass

    @abstractmethod
    def _get_row_oracle_matvec_fn(self) -> callable:
        """Return the kernel-specific row oracle matvec function.

        This should return a standalone function that implements the kernel-specific
        matrix-vector product for the row oracle.
        """
        pass

    def row_oracle(self, blk: torch.Tensor) -> DistributedLinOp:
        """Generic implementation of row oracle for all kernel types."""
        # Get the kernel-specific matvec function
        row_oracle_matvec_fn = self._get_row_oracle_matvec_fn()

        # Create operators for each device
        row_ops = []
        for device, A_chunk in enumerate(self.devices, self.A_chunks):
            # Create matvec function with kernel-specific implementation
            matvec_fn = partial(
                row_oracle_matvec_fn,
                A_mat=self.A_mat,
                row_idx=blk,
                A_chunk=A_chunk,
                kernel_params=self.kernel_params,
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
            is_new=False,
        )

    def blk_oracle(self, blk: torch.Tensor) -> SymmetricLinOp:
        """Get a symmetric operator for a block.

        Args:
            blk: Indices defining the block

        Returns:
            A symmetric linear operator for the specified block
        """
        blk_matvec = partial(self._blk_oracle_matvec, blk_idx=blk)

        return SymmetricLinOp(
            device=self.compute_device,
            shape=torch.Size((blk.shape[0], blk.shape[0])),
            matvec=blk_matvec,
            matmat=blk_matvec,
        )

    def shutdown(self):
        """Extend shutdown to clear caches."""
        # Clear kernel caches to free memory
        for op in self.kernel_ops:
            if hasattr(op, "_clear_cache"):
                op._clear_cache()

        # Clear the global caches
        global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
        _KERNEL_CACHE.clear()
        _LAZY_TENSOR_CACHE.clear()
        print(f"Cleared global caches on shutdown. PID: {os.getpid()}")

        # Call parent shutdown
        super().shutdown()
