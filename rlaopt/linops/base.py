from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.multiprocessing import Manager, Process, Queue, set_start_method

from .enums import _DistributionMode, _Operation
from rlaopt.utils import _is_list, _is_torch_device, _is_torch_f32_f64, _is_torch_size


class _BaseLinOp(ABC):
    """Base class for all linear operators."""

    def __init__(self, device: torch.device, shape: torch.Size, dtype: torch.dtype):
        self._check_inputs_base(device, shape, dtype)
        self._device = device
        self._shape = shape
        self._dtype = dtype

    def _check_inputs_base(self, device: Any, shape: Any, dtype: Any):
        _is_torch_device(device, "device")
        _is_torch_size(shape, "shape")
        if len(shape) != 2:
            raise ValueError(f"shape must have two elements. Received {len(shape)}")
        if not all(isinstance(i, int) and i > 0 for i in shape):
            raise ValueError(f"shape must contain positive integers. Received {shape}")
        _is_torch_f32_f64(dtype, "dtype")

    @property
    def device(self) -> torch.device:
        """Return the device of the linear operator."""
        return self._device

    @property
    def devices(self) -> list[torch.device]:
        """Return all devices used by the linear operator.

        For non-distributed linear operators, this is just a list containing the device
        of the operator. For distributed linear operators, this should return a list of
        devices containing the devices of each component linear operator.
        """
        return [self._device]

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def T(self) -> "_BaseLinOp":
        """Transpose of the linear operator.

        By default, linear operators don't support transposition. Subclasses like
        TwoSidedLinOp should override this property.
        """
        raise NotImplementedError("This linear operator doesn't support transposition")

    @abstractmethod
    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication."""
        pass

    def __rmatmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Right matrix-vector multiplication.

        By default, linear operators don't support right multiplication. Subclasses like
        TwoSidedLinOp should override this method.
        """
        raise NotImplementedError(
            "This linear operator doesn't support right multiplication"
        )


class _BaseDistributedLinOp(_BaseLinOp):
    """Base class for all distributed linear operators handling multiprocessing
    setup."""

    def __init__(
        self,
        shape: torch.Size,
        A: list[_BaseLinOp],
        distribution_mode: _DistributionMode,
        is_new=True,
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
    ):
        self._is_new = is_new
        if self._is_new:
            _is_list(A, "A")
            if not all(isinstance(i, _BaseLinOp) for i in A):
                raise ValueError("All elements in A must be instances of _BaseLinOp")
            # Check that all _BaseLinOp instances in A have the same dtype
            if not all(A_i.dtype == A[0].dtype for A_i in A):
                raise ValueError(
                    "All linear operators must have the same dtype. "
                    f"Received {', '.join(str(A_i.dtype) for A_i in A)}."
                )

        placeholder_device = torch.device(
            "cpu"
        )  # Only used for initializing the base class
        super().__init__(device=placeholder_device, shape=shape, dtype=A[0].dtype)

        # Store the list of linear operators and their devices
        self._A = A
        self._devices = [A_i.device for A_i in A]

        # Configure distribution via multiprocessing
        self._distribution_mode = _DistributionMode._from_str(
            distribution_mode, "distribution_mode"
        )

        if self._is_new:
            # Set the start method for multiprocessing
            set_start_method("spawn", force=True)

            # Create device-specific worker processes
            self._manager = Manager()
            self._result_queue = self._manager.Queue()
            self._task_queues = {}
            self._workers = {}

            # Start a dedicated worker for each device
            for device in self.devices:
                if device not in self._task_queues:
                    # For each device, create a task queue and a worker process
                    self._task_queues[device] = Queue()
                    worker = Process(
                        target=self._device_worker,
                        args=(device, self._task_queues[device], self._result_queue),
                        daemon=True,
                    )
                    worker.start()
                    self._workers[device] = worker
        else:
            # Reuse the existing manager, queues, and workers
            self._manager = manager
            self._result_queue = result_queue
            self._task_queues = task_queues
            self._workers = workers

    @property
    def device(self):
        """Raises an error to indicate that distributed operators don't have a single
        device.

        Use `devices` property instead to get all devices used by this operator.
        """
        raise AttributeError(
            "Distributed linear operators operate across multiple devices "
            "and don't have a single 'device'. "
            "Use the 'devices' property instead to get the list of all devices."
        )

    @property
    def devices(self):
        """Return the list of all devices used by this distributed operator."""
        return self._devices

    @staticmethod
    def _device_worker(
        device: torch.device,
        task_queue: Queue,
        result_queue: Queue,
    ):
        """Worker process that handles tasks for a specific device."""
        while True:
            # Get the task from the queue
            task = task_queue.get()
            if task is None:  # Shutdown signal
                break

            task_id, linop, x, operation = task
            x = x.to(device)

            # Perform the operation
            try:
                if operation == _Operation.MATVEC:
                    result = linop @ x
                elif operation == _Operation.RMATVEC:
                    result = linop.T @ x
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                result_queue.put((task_id, result.cpu()))
            except Exception as e:
                # Handle exceptions and send them back to the main process
                result_queue.put((task_id, e))

    def _chunk_tensor(self, x: torch.Tensor, by_dimension: int) -> list[torch.Tensor]:
        """Chunk the input tensor into smaller tensors according to the desired
        dimension."""
        x_chunks = []
        start_idx = 0
        for i in range(len(self._A)):
            end_idx = start_idx + self._A[i].shape[by_dimension]
            x_chunks.append(x[start_idx:end_idx].cpu())
            start_idx = end_idx
        return x_chunks

    def _distribute_tasks(
        self, x: torch.Tensor, operation: _Operation, chunk: bool, by_dimension: int
    ):
        """Distribute tasks to the worker processes."""
        # Decided whether or not to chunk the input tensor
        if chunk:
            x_chunks = self._chunk_tensor(x, by_dimension)
            # Dispatch chunked tasks to the workers
            for i, (linop, device, x_chunk) in enumerate(
                zip(self._A, self._devices, x_chunks)
            ):
                self._task_queues[device].put((i, linop, x_chunk, operation))
        else:
            # Dispatch the entire tensor to all workers
            for i, (linop, device) in enumerate(zip(self._A, self._devices)):
                self._task_queues[device].put((i, linop, x.cpu(), operation))

    def _gather_results(self, num_tasks: int) -> list[torch.Tensor]:
        """Gather results from the worker processes."""
        results = [None] * num_tasks
        for _ in range(num_tasks):
            task_id, result = self._result_queue.get()
            if isinstance(result, Exception):
                raise RuntimeError(f"Error occurred in worker process: {result}")
            results[task_id] = result
        return results

    def _combine_results(
        self, results: list[torch.Tensor], concatenate: bool
    ) -> torch.Tensor:
        """Combine the results from all workers either by concatenation of summation."""
        if concatenate:
            return torch.cat(results, dim=0)
        else:
            return sum(results)

    def shutdown(self):
        """Shut down worker processes."""
        # Only shut down if we own the processes
        if self._is_new:
            for _, queue in self._task_queues.items():
                queue.put(None)  # Signal worker to exit

            for _, worker in self._workers.items():
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()

    def __del__(self):
        # Clean up workers
        self.shutdown()
