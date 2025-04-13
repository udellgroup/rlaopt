from functools import partial
import os
from typing import Any, Callable, Dict, Optional, Set

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    SymmetricLinOp,
    DistributedLinOp,
    DistributedSymmetricLinOp,
)
from rlaopt.linops.distributed import _DistributedLinOp
from rlaopt.utils import _is_torch_tensor, _is_dict, _is_set

# Global, module-level cache to persist across worker calls
_KERNEL_CACHE: Dict[str, LazyTensor] = {}
_LAZY_TENSOR_CACHE: Dict[str, LazyTensor] = {}


class _KernelLinOp(SymmetricLinOp):
    def __init__(
        self,
        A: torch.Tensor,
        kernel_params: Dict[str, Any],
        _check_kernel_params_fn: Callable,
        _kernel_computation_fn: Callable,
    ):
        self._check_kernel_params = _check_kernel_params_fn
        self._check_inputs(A, kernel_params)
        self._A = A
        self._kernel_params = kernel_params
        self._kernel_computation = _kernel_computation_fn
        self._K_lazy = self._get_kernel()
        super().__init__(
            device=self._A.device,
            shape=torch.Size((self._A.shape[0], self._A.shape[0])),
            matvec=lambda x: self._K_lazy @ x,
            matmat=lambda x: self._K_lazy @ x,
            dtype=self._A.dtype,
        )

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def kernel_params(self) -> Dict[str, Any]:
        return self._kernel_params

    def _check_inputs(self, A: Any, kernel_params: Any):
        _is_torch_tensor(A, "A")
        if A.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got {A.ndim}D tensor.")
        _is_dict(kernel_params, "kernel_params")
        self._check_kernel_params(kernel_params)

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

        K_lazy = self._kernel_computation(Ai_lazy, Aj_lazy, self.kernel_params)
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
            dtype=self.dtype,
        )

    def row_oracle(self, blk: torch.Tensor):
        return self._get_kernel_linop(idx1=blk, symmetric=False)

    def blk_oracle(self, blk: torch.Tensor):
        return self._get_kernel_linop(idx1=blk, idx2=blk, symmetric=True)


class _CacheableKernelLinOp(TwoSidedLinOp):
    """Private implementation of Kernel linear operator with caching."""

    def __init__(
        self,
        A: torch.Tensor,
        kernel_params: Dict[str, Any],
        chunk_idx: torch.Tensor,
        device: torch.device,
        _kernel_computation_fn: Callable,
        _kernel_name: str,
    ):
        self._A = A.to(device)
        self._kernel_params = kernel_params
        self._chunk_idx = chunk_idx
        self._kernel_computation = _kernel_computation_fn
        self._unique_id = (
            f"{_kernel_name}_{id(self)}_{len(A)}_{kernel_params}_{A.device}"
        )

        super().__init__(
            device=device,
            shape=torch.Size((self._chunk_idx.shape[0], self._A.shape[0])),
            matvec=self._matvec,
            rmatvec=self._rmatvec,
            matmat=self._matvec,
            rmatmat=self._rmatvec,
            dtype=self._A.dtype,
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

    def _get_lazy_tensors(self):
        Ab_lazy = LazyTensor(self.A[self.chunk_idx][:, None, :])
        A_lazy = LazyTensor(self.A[None, :, :])
        return Ab_lazy, A_lazy

    def _get_kernel(self):
        """Get the cached kernel or compute it if not present."""
        global _KERNEL_CACHE

        # Use process ID to ensure cache is per-process
        pid = os.getpid()
        cache_key = f"{pid}_{self._unique_id}"

        if cache_key not in _KERNEL_CACHE:
            print(f"[PID {pid}] Computing kernel for device {self.device}...")

            # Compute kernel and store in the global cache
            Ab_lazy, A_lazy = self._get_lazy_tensors()
            _KERNEL_CACHE[cache_key] = self._kernel_computation(
                Ab_lazy, A_lazy, self.kernel_params
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

    # print(f"[PID {pid}] Cache key: {cache_key}")

    # if cache_key not in _LAZY_TENSOR_CACHE:
    #     _LAZY_TENSOR_CACHE[cache_key] = LazyTensor(A[None, :, :])
    #     print(f"[PID {pid}] Created new LazyTensor for device {A.device}")
    # else:
    #     print(f"[PID {pid}] Using cached LazyTensor for device {A.device}")

    if cache_key not in _LAZY_TENSOR_CACHE:
        _LAZY_TENSOR_CACHE[cache_key] = LazyTensor(A[None, :, :])

    return _LAZY_TENSOR_CACHE[cache_key]


# class _DistributedKernelLinOp(DistributedSymmetricLinOp):
#     """Abstract base class for distributed kernel linear operators."""

#     def __init__(
#         self,
#         A: torch.Tensor,
#         kernel_params: Dict[str, Any],
#         devices: Set[torch.device],
#         compute_device: Optional[torch.device],
#         _check_kernel_params_fn: Callable,
#         _kernel_computation_fn: Callable,
#         _row_oracle_matvec_fn: Callable,
#         _cacheable_kernel_name: str,
#     ):
#         """Initialize the distributed kernel linear operator.

#         Args:
#             A: Input data tensor
#             kernel_params: Dictionary of kernel parameters
#             devices: Set of devices to distribute computation across
#             compute_device: Device to use for block computation
#             (default: first device in devices)
#         """
#         # Clean the global caches at initialization
#         global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
#         _KERNEL_CACHE.clear()
#         _LAZY_TENSOR_CACHE.clear()
#         print(f"Initialized with clean caches. PID: {os.getpid()}")

#         # Save parameters
#         self._check_kernel_params = _check_kernel_params_fn
#         self._check_inputs(A, kernel_params, devices, compute_device)
#         self._kernel_computation = _kernel_computation_fn
#         self._row_oracle_matvec = _row_oracle_matvec_fn
#         self._cacheable_kernel_name = _cacheable_kernel_name
#         self._A_mat = A  # Keep original tensor for oracles
#         self._kernel_params = kernel_params
#         self.devices = list(devices)
#         self.compute_device = compute_device or self.devices[0]
#         self._kernel_params_devices = self._get_kernel_params_devices()

#         # Create row partitioning
#         self.A_row_chunks = torch.chunk(
#             torch.arange(self._A_mat.shape[0]), len(self.devices), dim=0
#         )

#         # Create chunks of data for each device
#         self.A_chunks = []
#         for device, chunk_idx in zip(self.devices, self.A_row_chunks):
#             # We keep chunks on each device for row_oracle
#             self.A_chunks.append(self._A_mat[chunk_idx].to(device))

#         # Create cacheable kernel operators for each chunk
#         kernel_ops = self._create_kernel_operators()

#         # Initialize the distributed operator
#         super().__init__(
#             shape=torch.Size((self._A_mat.shape[0], self._A_mat.shape[0])),
#             A=kernel_ops,
#             distribution_mode="row",
#         )

#         # Store references for cleanup
#         self.kernel_ops = kernel_ops

#     @property
#     def A_mat(self) -> torch.Tensor:
#         return self._A_mat

#     @property
#     def kernel_params(self) -> Dict[str, Any]:
#         return self._kernel_params

#     def _check_inputs(
#         self,
#         A: Any,
#         kernel_params: Any,
#         devices: Any,
#         compute_device: Any,
#     ):
#         _is_torch_tensor(A, "A")
#         if A.ndim != 2:
#             raise ValueError(f"A must be a 2D tensor, got {A.ndim}D tensor.")
#         _is_dict(kernel_params, "kernel_params")
#         self._check_kernel_params(kernel_params)
#         _is_set(devices, "devices")
#         if len(devices) == 0:
#             raise ValueError("devices must be a non-empty set.")
#         if not all(isinstance(d, torch.device) for d in devices):
#             raise ValueError("All elements in devices "
#                               "must be torch.device instances.")
#         if compute_device is not None:
#             _is_torch_device(compute_device, "compute_device")
#             if compute_device not in devices:
#                 raise ValueError("compute_device must be in the set of devices.")

#     def _get_kernel_params_devices(self):
#         """Move kernel parameters to the devices.

#         Returns:
#             Dictionary of kernel parameters moved to the devices
#         """
#         kernel_params_devices = {}
#         for device in self.devices:
#             kernel_params_devices[device] = self._kernel_params.copy()
#             for key, value in self._kernel_params.items():
#                 if isinstance(value, torch.Tensor):
#                     kernel_params_devices[device][key] = value.to(device)
#                 else:
#                     kernel_params_devices[device][key] = value
#         return kernel_params_devices

#     def _create_kernel_operators(self):
#         """Create the kernel operators for each chunk.

#         Returns:
#             List of kernel operators, one for each device/chunk
#         """
#         ops = []
#         for device, chunk_idx in zip(self.devices, self.A_row_chunks):
#             ops.append(
#                 _CacheableKernelLinOp(
#                     A=self.A_mat,
#                     kernel_params=self._kernel_params_devices[device],
#                     chunk_idx=chunk_idx,
#                     device=device,
#                     _kernel_computation_fn=self._kernel_computation,
#                     _kernel_name=self._cacheable_kernel_name,
#                 )
#             )
#         return ops

#     def row_oracle(self, blk: torch.Tensor) -> DistributedLinOp:
#         """Generic implementation of row oracle for all kernel types."""
#         # Create operators for each device
#         row_ops = []
#         for device, A_chunk in zip(self.devices, self.A_chunks):
#             # Create matvec function with kernel-specific implementation
#             matvec_fn = partial(
#                 self._row_oracle_matvec,
#                 A_mat=self.A_mat,
#                 row_idx=blk,
#                 A_chunk=A_chunk,
#                 kernel_params=self._kernel_params_devices[device],
#                 kernel_computation=self._kernel_computation,
#             )

#             # Create a LinOp with the matvec function
#             row_ops.append(
#                 LinOp(
#                     device=device,
#                     shape=torch.Size((blk.shape[0], A_chunk.shape[0])),
#                     matvec=matvec_fn,
#                     matmat=matvec_fn,
#                 )
#             )

#         # Create a distributed operator that reuses our workers
#         return _DistributedLinOp(
#             shape=torch.Size((blk.shape[0], self.A_mat.shape[0])),
#             A=row_ops,
#             distribution_mode="column",
#             manager=self._manager,
#             result_queue=self._result_queue,
#             task_queues=self._task_queues,
#             workers=self._workers,
#             is_new=False,
#         )

#     def _get_blk_lazy_tensors(self, blk: torch.Tensor)
# -> tuple[LazyTensor, LazyTensor]:
#         """Get LazyTensor representations for the block.

#         Args:
#             blk: Indices defining the block
#         Returns:
#             Tuple of LazyTensors for the block
#         """
#         A_blk = self.A_mat[blk].to(self.compute_device)
#         Abi_lazy = LazyTensor(A_blk[:, None, :])
#         Abj_lazy = LazyTensor(A_blk[None, :, :])
#         return Abi_lazy, Abj_lazy

#     def _blk_oracle_matvec(self, x: torch.Tensor, blk: torch.Tensor) -> torch.Tensor:
#         """Compute kernel matrix-vector product for block oracle."""
#         Abi_lazy, Abj_lazy = self._get_blk_lazy_tensors(blk)
#         K_lazy = self._kernel_computation(Abi_lazy, Abj_lazy, self.kernel_params)
#         return K_lazy @ x

#     def blk_oracle(self, blk: torch.Tensor) -> SymmetricLinOp:
#         """Get a symmetric operator for a block.

#         Args:
#             blk: Indices defining the block

#         Returns:
#             A symmetric linear operator for the specified block
#         """
#         # Get the lazy tensors for the block
#         blk_matvec = partial(self._blk_oracle_matvec, blk=blk)

#         return SymmetricLinOp(
#             device=self.compute_device,
#             shape=torch.Size((blk.shape[0], blk.shape[0])),
#             matvec=blk_matvec,
#             matmat=blk_matvec,
#             dtype=self.A_mat.dtype,
#         )

#     def shutdown(self):
#         """Extend shutdown to clear caches."""
#         # Clear kernel caches to free memory
#         for op in self.kernel_ops:
#             if hasattr(op, "_clear_cache"):
#                 op._clear_cache()

#         # Clear the global caches
#         global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
#         _KERNEL_CACHE.clear()
#         _LAZY_TENSOR_CACHE.clear()
#         print(f"Cleared global caches on shutdown. PID: {os.getpid()}")

#         # Call parent shutdown
#         super().shutdown()


class _DistributedKernelLinOp(DistributedSymmetricLinOp):
    """Abstract base class for distributed kernel linear operators."""

    def __init__(
        self,
        A: torch.Tensor,
        kernel_params: Dict[str, Any],
        devices: Set[torch.device],
        _check_kernel_params_fn: Callable,
        _kernel_computation_fn: Callable,
        _row_oracle_matvec_fn: Callable,
        _cacheable_kernel_name: str,
    ):
        """Initialize the distributed kernel linear operator.

        Args:
            A: Input data tensor
            kernel_params: Dictionary of kernel parameters
            devices: Set of devices to distribute computation across
        """
        # Clean the global caches at initialization
        global _KERNEL_CACHE, _LAZY_TENSOR_CACHE
        _KERNEL_CACHE.clear()
        _LAZY_TENSOR_CACHE.clear()
        print(f"Initialized with clean caches. PID: {os.getpid()}")

        # Save parameters
        self._check_kernel_params = _check_kernel_params_fn
        self._check_inputs(A, kernel_params, devices)
        self._kernel_computation = _kernel_computation_fn
        self._row_oracle_matvec = _row_oracle_matvec_fn
        self._cacheable_kernel_name = _cacheable_kernel_name
        self._A_mat = A  # Keep original tensor for oracles
        self._kernel_params = kernel_params
        self.devices = list(devices)
        self._kernel_params_devices = self._get_kernel_params_devices()

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
            distribution_mode="row",
        )

        # Store references for cleanup
        self.kernel_ops = kernel_ops

    @property
    def A_mat(self) -> torch.Tensor:
        return self._A_mat

    @property
    def kernel_params(self) -> Dict[str, Any]:
        return self._kernel_params

    def _check_inputs(
        self,
        A: Any,
        kernel_params: Any,
        devices: Any,
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

    def _get_kernel_params_devices(self):
        """Move kernel parameters to the devices.

        Returns:
            Dictionary of kernel parameters moved to the devices
        """
        kernel_params_devices = {}
        for device in self.devices:
            kernel_params_devices[device] = self._kernel_params.copy()
            for key, value in self._kernel_params.items():
                if isinstance(value, torch.Tensor):
                    kernel_params_devices[device][key] = value.to(device)
                else:
                    kernel_params_devices[device][key] = value
        return kernel_params_devices

    def _create_kernel_operators(self):
        """Create the kernel operators for each chunk.

        Returns:
            List of kernel operators, one for each device/chunk
        """
        ops = []
        for device, chunk_idx in zip(self.devices, self.A_row_chunks):
            ops.append(
                _CacheableKernelLinOp(
                    A=self.A_mat,
                    kernel_params=self._kernel_params_devices[device],
                    chunk_idx=chunk_idx,
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
        for device, A_chunk in zip(self.devices, self.A_chunks):
            # Create matvec function with kernel-specific implementation
            matvec_fn = partial(
                self._row_oracle_matvec,
                A_mat=self.A_mat,
                row_idx=blk,
                A_chunk=A_chunk,
                kernel_params=self._kernel_params_devices[device],
                kernel_computation=self._kernel_computation,
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
            distribution_mode="column",
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
            is_new=False,
        )

    def blk_oracle(self, blk: torch.Tensor) -> DistributedSymmetricLinOp:
        """Get a distributed symmetric operator for a block.

        Args:
            blk: Indices defining the block

        Returns:
            A distributed symmetric linear operator for the specified block
        """
        # Create a list of operators, one for each device
        block_ops = []

        # Split the block indices into chunks for each device
        blk_chunks = torch.chunk(torch.arange(blk.shape[0]), len(self.devices), dim=0)

        # Create an operator for each device and chunk
        for device, blk_chunk_idx in zip(self.devices, blk_chunks):
            # Get the actual indices for this chunk
            blk_chunk = blk[blk_chunk_idx]

            # Create a matvec function for this chunk
            def chunk_matvec(x, _device=device, _blk_chunk=blk_chunk, _blk=blk):
                # Move input to the device
                x = x.to(_device)

                # Get the data for this chunk
                A_blk_chunk = self.A_mat[_blk_chunk].to(_device)
                A_blk_full = self.A_mat[_blk].to(_device)

                # Create LazyTensors
                Abi_lazy = LazyTensor(A_blk_chunk[:, None, :])
                Abj_lazy = LazyTensor(A_blk_full[None, :, :])

                # Compute kernel and matrix-vector product
                K_lazy = self._kernel_computation(
                    Abi_lazy, Abj_lazy, self._kernel_params_devices[_device]
                )
                return K_lazy @ x

            # Create a linear operator for this chunk
            block_ops.append(
                LinOp(
                    device=device,
                    shape=torch.Size((blk_chunk_idx.shape[0], blk.shape[0])),
                    matvec=chunk_matvec,
                    matmat=chunk_matvec,
                    dtype=self.A_mat.dtype,
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
