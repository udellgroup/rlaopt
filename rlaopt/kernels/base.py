from functools import partial
import os
from typing import Any, Callable, Optional, Set

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    DistributedLinOp,
    DistributedTwoSidedLinOp,
)
from rlaopt.linops.distributed import _DistributedLinOp
from rlaopt.utils import _is_torch_tensor, _is_set
from .configs import KernelConfig, _is_kernel_config

# Global, module-level cache to persist across worker calls
_KERNEL_CACHE: dict[str, LazyTensor] = {}
_LAZY_TENSOR_CACHE: dict[str, LazyTensor] = {}


class _KernelLinOp(TwoSidedLinOp):
    def __init__(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        kernel_config: KernelConfig,
        _kernel_computation_fn: Callable,
    ):
        self._check_inputs(A1, A2, kernel_config)
        self._A1 = A1
        self._A2 = A2
        self._kernel_config = kernel_config
        self._kernel_computation = _kernel_computation_fn
        self._K_lazy = self._get_kernel()
        super().__init__(
            device=self._A1.device,
            shape=torch.Size((self._A1.shape[0], self._A2.shape[0])),
            matvec=lambda x: self._K_lazy @ x,
            rmatvec=lambda x: self._K_lazy.T @ x,
            matmat=lambda x: self._K_lazy @ x,
            rmatmat=lambda x: self._K_lazy.T @ x,
            dtype=self._A1.dtype,
        )

    @property
    def A1(self) -> torch.Tensor:
        return self._A1

    @property
    def A2(self) -> torch.Tensor:
        return self._A2

    @property
    def kernel_config(self) -> KernelConfig:
        return self._kernel_config

    def _check_inputs(self, A1: Any, A2, kernel_config: Any):
        _is_torch_tensor(A1, "A1")
        _is_torch_tensor(A2, "A2")
        if A1.ndim != 2:
            raise ValueError(f"A1 must be a 2D tensor, got {A1.ndim}D tensor.")
        if A2.ndim != 2:
            raise ValueError(f"A2 must be a 2D tensor, got {A2.ndim}D tensor.")
        if A1.device != A2.device:
            raise ValueError("A1 and A2 must be on the same device.")
        if A1.dtype != A2.dtype:
            raise ValueError("A1 and A2 must have the same dtype.")
        _is_kernel_config(kernel_config, "kernel_config")

    def _get_kernel(
        self, idx1: Optional[torch.Tensor] = None, idx2: Optional[torch.Tensor] = None
    ) -> LazyTensor:
        if idx1 is None:
            A1_lazy = LazyTensor(self.A1[:, None, :])
        else:
            A1_lazy = LazyTensor(self.A1[idx1][:, None, :])

        if idx2 is None:
            A2_lazy = LazyTensor(self.A2[None, :, :])
        else:
            A2_lazy = LazyTensor(self.A2[idx2][None, :, :])

        K_lazy = self._kernel_computation(A1_lazy, A2_lazy, self.kernel_config)
        return K_lazy

    def _get_kernel_linop(
        self,
        idx1: Optional[torch.Tensor] = None,
        idx2: Optional[torch.Tensor] = None,
    ) -> LinOp:
        K = self._get_kernel(idx1, idx2)
        return LinOp(
            device=self.device,
            shape=torch.Size(K.shape),
            matvec=lambda x: K @ x,
            matmat=lambda x: K @ x,
            dtype=self.dtype,
        )

    def row_oracle(self, blk: torch.Tensor) -> LinOp:
        return self._get_kernel_linop(idx1=blk)

    def blk_oracle(self, blk: torch.Tensor) -> LinOp:
        return self._get_kernel_linop(idx1=blk, idx2=blk)


class _CacheableKernelLinOp(TwoSidedLinOp):
    """Private implementation of Kernel linear operator with caching."""

    def __init__(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        kernel_config: KernelConfig,
        device: torch.device,
        _kernel_computation_fn: Callable,
        _kernel_name: str,
    ):
        self._A1 = A1.to(device)
        self._A2 = A2.to(device)
        self._kernel_config = kernel_config
        self._kernel_computation = _kernel_computation_fn
        self._unique_id = (
            f"{_kernel_name}_{id(self)}_{len(self._A1)}_{len(self._A2)}_"
            f"{self._A1.device}"
        )
        super().__init__(
            device=device,
            shape=torch.Size((self._A1.shape[0], self._A2.shape[0])),
            matvec=self._matvec,
            rmatvec=self._rmatvec,
            matmat=self._matvec,
            rmatmat=self._rmatvec,
            dtype=self._A1.dtype,
        )

    @property
    def A1(self) -> torch.Tensor:
        return self._A1

    @property
    def A2(self) -> torch.Tensor:
        return self._A2

    @property
    def kernel_config(self) -> KernelConfig:
        return self._kernel_config

    def _get_lazy_tensors(self):
        A1_lazy = LazyTensor(self.A1[:, None, :])
        A2_lazy = LazyTensor(self.A2[None, :, :])
        return A1_lazy, A2_lazy

    def _get_kernel(self):
        """Get the cached kernel or compute it if not present."""
        global _KERNEL_CACHE

        # Use process ID to ensure cache is per-process
        pid = os.getpid()
        cache_key = f"{pid}_{self._unique_id}"

        if cache_key not in _KERNEL_CACHE:
            print(f"[PID {pid}] Computing kernel for device {self.device}...")

            # Compute kernel and store in the global cache
            A1_lazy, A2_lazy = self._get_lazy_tensors()
            _KERNEL_CACHE[cache_key] = self._kernel_computation(
                A1_lazy, A2_lazy, self.kernel_config
            )

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
    """Get a cached LazyTensor or create a new one.

    This is used in the row oracle to avoid recomputing the LazyTensor for each call.
    The input is one of the chunks of the original matrix.
    """
    global _LAZY_TENSOR_CACHE

    pid = os.getpid()
    # Use data_ptr() to ensure unique key for each tensor
    # A.shape and A.device are also added for safety
    cache_key = f"{pid}_lazy_{A.data_ptr()}_{A.shape}_{A.device}"

    if cache_key not in _LAZY_TENSOR_CACHE:
        _LAZY_TENSOR_CACHE[cache_key] = LazyTensor(A[None, :, :])

    return _LAZY_TENSOR_CACHE[cache_key]


class _DistributedKernelLinOp(DistributedTwoSidedLinOp):
    """Abstract base class for distributed kernel linear operators."""

    def __init__(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        kernel_config: KernelConfig,
        devices: Set[torch.device],
        _kernel_computation_fn: Callable,
        _row_oracle_matvec_fn: Callable,
        _block_chunk_matvec_fn: Callable,
        _cacheable_kernel_name: str,
    ):
        """Initialize the distributed kernel linear operator.

        Args:
            A1: Input data tensor
            A2: Input data tensor
            kernel_config: Kernel configuration
            devices: Set of devices to distribute computation across
        """
        # Clean the global caches at initialization
        global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
        _KERNEL_CACHE.clear()
        _LAZY_TENSOR_CACHE.clear()
        print(f"Initialized with clean caches. PID: {os.getpid()}")

        # Save parameters
        self._check_inputs(A1, A2, kernel_config, devices)
        self._kernel_computation = _kernel_computation_fn
        self._row_oracle_matvec = _row_oracle_matvec_fn
        self._block_chunk_matvec = _block_chunk_matvec_fn
        self._cacheable_kernel_name = _cacheable_kernel_name
        self._A1 = A1  # Keep original tensor for oracles
        self._A2 = A2  # Keep original tensor for oracles
        self._kernel_config = kernel_config
        self.devices = list(devices)
        self._kernel_config_devices = [
            self._kernel_config.to(device) for device in devices
        ]

        # Create row partitioning
        # A1_row_chunks is useful for the linop and block oracle,
        # A2_row_chunks is useful for the row oracle
        self.A1_row_chunks = torch.chunk(
            torch.arange(self._A1.shape[0]), len(self.devices), dim=0
        )
        self.A2_row_chunks = torch.chunk(
            torch.arange(self._A2.shape[0]), len(self.devices), dim=0
        )

        # Create chunks of data for each device
        self.A1_chunks = []
        self.A2_chunks = []
        for device, chunk_idx in zip(self.devices, self.A1_row_chunks):
            self.A1_chunks.append(self._A1[chunk_idx].to(device))
        for device, chunk_idx in zip(self.devices, self.A2_row_chunks):
            self.A2_chunks.append(self._A2[chunk_idx].to(device))

        # Create cacheable kernel operators for each chunk
        kernel_ops = self._create_kernel_operators()

        # Initialize the distributed operator
        super().__init__(
            shape=torch.Size((self._A1.shape[0], self._A2.shape[0])),
            A=kernel_ops,
            distribution_mode="row",
        )

        # Store references for cleanup
        self.kernel_ops = kernel_ops

    @property
    def A1(self) -> torch.Tensor:
        return self._A1

    @property
    def A2(self) -> torch.Tensor:
        return self._A2

    @property
    def kernel_config(self) -> KernelConfig:
        return self._kernel_config

    def _check_inputs(
        self,
        A1: Any,
        A2: Any,
        kernel_config: Any,
        devices: Any,
    ):
        _is_torch_tensor(A1, "A1")
        _is_torch_tensor(A2, "A2")
        if A1.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got {A1.ndim}D tensor.")
        if A2.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got {A2.ndim}D tensor.")
        if A1.dtype != A2.dtype:
            raise ValueError("A1 and A2 must have the same dtype.")
        _is_kernel_config(kernel_config, "kernel_config")
        _is_set(devices, "devices")
        if len(devices) == 0:
            raise ValueError("devices must be a non-empty set.")
        if not all(isinstance(d, torch.device) for d in devices):
            raise ValueError("All elements in devices must be torch.device instances.")

    def _create_kernel_operators(self):
        """Create the kernel operators for each chunk.

        Returns:
            List of kernel operators, one for each device/chunk
        """
        ops = []
        for device, chunk_idx in zip(self.devices, self.A1_row_chunks):
            ops.append(
                _CacheableKernelLinOp(
                    A1=self.A1[chunk_idx],
                    A2=self.A2,
                    kernel_config=self._kernel_config_devices[device],
                    device=device,
                    _kernel_computation_fn=self._kernel_computation,
                    _kernel_name=self._cacheable_kernel_name,
                )
            )
        return ops

    def row_oracle(self, blk: torch.Tensor) -> DistributedLinOp:
        """Generic implementation of row oracle for all kernel types."""
        # Create operators for each device
        row_ops = []
        for device, A2_chunk in zip(self.devices, self.A2_chunks):
            # Create matvec function with kernel-specific implementation
            matvec_fn = partial(
                self._row_oracle_matvec,
                A1=self.A1,
                A2_chunk=A2_chunk,
                blk=blk,
                kernel_config=self._kernel_config_devices[device],
                kernel_computation=self._kernel_computation,
            )

            # Create a LinOp with the matvec function
            row_ops.append(
                LinOp(
                    device=device,
                    shape=torch.Size((blk.shape[0], A2_chunk.shape[0])),
                    matvec=matvec_fn,
                    matmat=matvec_fn,
                    dtype=self.dtype,
                )
            )

        # Create a distributed operator that reuses our workers
        return _DistributedLinOp(
            shape=torch.Size((blk.shape[0], self.A2.shape[0])),
            A=row_ops,
            distribution_mode="column",
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
            is_new=False,
        )

    def blk_oracle(self, blk: torch.Tensor) -> DistributedLinOp:
        """Get a distributed linear operator for a block.

        Args:
            blk: Indices defining the block

        Returns:
            A distributed linear operator for the specified block
        """
        # Create a list of operators, one for each device
        block_ops = []

        # Split the block indices into chunks for each device
        blk_chunks = torch.chunk(torch.arange(blk.shape[0]), len(self.devices), dim=0)

        # Create an operator for each device and chunk
        for device, blk_chunk_idx in zip(self.devices, blk_chunks):
            # Get the actual indices for this chunk
            blk_chunk = blk[blk_chunk_idx]

            # Create a matvec function for this chunk using partial
            matvec_fn = partial(
                self._block_chunk_matvec,
                device=device,
                A1=self.A1,
                A2=self.A2,
                blk_chunk=blk_chunk,
                blk=blk,
                kernel_config=self._kernel_config_devices[device],
                kernel_computation=self._kernel_computation,
            )

            # Create a linear operator for this chunk
            block_ops.append(
                LinOp(
                    device=device,
                    shape=torch.Size((blk_chunk_idx.shape[0], blk.shape[0])),
                    matvec=matvec_fn,
                    matmat=matvec_fn,
                    dtype=self.dtype,
                )
            )

        # Create a distributed operator that reuses our workers
        return _DistributedLinOp(
            shape=torch.Size((blk.shape[0], blk.shape[0])),
            A=block_ops,
            distribution_mode="row",
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
            is_new=False,
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
