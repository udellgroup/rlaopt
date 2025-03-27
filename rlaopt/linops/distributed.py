from enum import Enum, auto
from typing import List

import torch
from torch.multiprocessing import Manager, Process, Queue, set_start_method

from rlaopt.utils import _is_list
from rlaopt.linops.base import _BaseLinOp
from rlaopt.linops.simple import LinOp, TwoSidedLinOp


__all__ = [
    "DistributionMode",
    "DistributedLinOp",
    "DistributedTwoSidedLinOp",
    "DistributedSymmetricLinOp",
]


class _Operation(Enum):
    MATVEC = auto()
    RMATVEC = auto()


class DistributionMode(Enum):
    ROW = auto()  # Matrix is distributed across rows
    COLUMN = auto()  # Matrix is distributed across columns

    @classmethod
    def from_string(cls, value):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.upper()
            if value == "ROW":
                return cls.ROW
            elif value == "COLUMN":
                return cls.COLUMN

        raise ValueError(
            f"Invalid distribution mode: {value}. "
            "Expected 'row', 'column', DistributionMode.ROW, "
            "or DistributionMode.COLUMN."
        )


class _DistributedLinOp(_BaseLinOp):
    """Base class with implementation details for distributed linear operators."""

    def __init__(
        self,
        shape: torch.Size,
        A: List[LinOp],
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
        is_new=True,
        distribution_mode=DistributionMode.ROW,
    ):
        super().__init__(shape=shape)
        self._is_new = is_new
        self._distribution_mode = DistributionMode.from_string(distribution_mode)

        # Validate input
        if self._is_new:
            _is_list(A, "A")
            if not all(isinstance(A_i, LinOp) for A_i in A):
                raise ValueError("All elements of A must be linear operators.")

        self._A = A

        if self._is_new:
            # Set the start method for multiprocessing
            set_start_method("spawn", force=True)

            # Create device-specific worker processes
            self._manager = Manager()
            self._result_queue = self._manager.Queue()
            self._task_queues = {}
            self._workers = {}

            # Start a dedicated worker for each device
            for linop in self._A:
                device = linop.device
                if device not in self._task_queues:
                    self._task_queues[device] = Queue()
                    worker = Process(
                        target=self._device_worker,
                        args=(device, self._task_queues[device], self._result_queue),
                    )
                    worker.daemon = True
                    worker.start()
                    self._workers[device] = worker
        else:
            # Use shared resources
            self._manager = manager
            self._result_queue = result_queue
            self._task_queues = task_queues
            self._workers = workers

    @staticmethod
    def _device_worker(device, task_queue, result_queue):
        """Worker process that handles tasks for a specific device."""
        if str(device).startswith("cuda"):
            device_id = int(str(device).split(":")[1])
            torch.cuda.set_device(device_id)

        while True:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                break

            task_id, linop, x, operation = task
            x = x.to(device)

            try:
                if operation == _Operation.MATVEC:
                    result = linop @ x
                elif operation == _Operation.RMATVEC:
                    # Check if this linear operator supports transpose operations
                    if hasattr(linop, "T"):
                        result = linop.T @ x
                    else:
                        raise AttributeError(
                            "Linear operator does not support transpose operations."
                        )
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                result_queue.put((task_id, result.cpu()))
            except Exception as e:
                # Send back the error so the main process can handle it
                result_queue.put((task_id, e))

    def _chunk_vector(self, w: torch.Tensor, by_dimension=0):
        """Split vector according to specified dimension of operators."""
        w_chunks = []
        start_idx = 0
        for i in range(len(self._A)):
            end_idx = start_idx + self._A[i].shape[by_dimension]
            w_chunks.append(w[start_idx:end_idx].cpu())
            start_idx = end_idx
        return w_chunks

    def _distribute_tasks(
        self, w: torch.Tensor, operation: _Operation, chunk: bool, by_dimension: int = 0
    ):
        """Common code for distributing tasks to workers."""
        # Decide whether to chunk or send full vector
        if chunk:
            chunks = self._chunk_vector(w, by_dimension)
            # Dispatch chunked tasks
            for i, (linop, w_chunk) in enumerate(zip(self._A, chunks)):
                self._task_queues[linop.device].put((i, linop, w_chunk, operation))
        else:
            # Send full vector to all workers
            for i, linop in enumerate(self._A):
                self._task_queues[linop.device].put((i, linop, w.cpu(), operation))

        # Collect results
        results = [None] * len(self._A)
        for _ in range(len(self._A)):
            task_id, result = self._result_queue.get()

            if isinstance(result, Exception):
                raise RuntimeError(f"Error in worker process: {result}")

            results[task_id] = result

        return results

    def _combine_results(self, results, w, concatenate=True):
        """Combine results either by concatenation or summation."""
        if concatenate:
            combined = torch.cat(results, dim=0)
        else:
            combined = sum(results)

        return combined.to(w.device)

    def _matvec(self, w: torch.Tensor):
        if self._distribution_mode == DistributionMode.ROW:
            # Row-distributed operator: send full vector, concatenate results
            results = self._distribute_tasks(w, _Operation.MATVEC, chunk=False)
            return self._combine_results(results, w, concatenate=True)
        else:  # COLUMN mode
            # Column-distributed operator: chunk by columns, sum results
            results = self._distribute_tasks(
                w, _Operation.MATVEC, chunk=True, by_dimension=1
            )
            return self._combine_results(results, w, concatenate=False)

    def _matmat(self, w: torch.Tensor):
        return self._matvec(w)

    def __matmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            return self._matvec(x)
        elif x.ndim == 2:
            return self._matmat(x)
        else:
            raise ValueError(f"x must be a 1D or 2D tensor. Received {x.ndim}D tensor.")

    def __del__(self):
        # Clean up workers
        self.shutdown()

    def shutdown(self):
        """Shut down worker processes."""
        # Only shut down if we own the processes
        if self._is_new:
            for device, queue in self._task_queues.items():
                queue.put(None)  # Signal worker to exit

            for device, worker in self._workers.items():
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()


class _DistributedTwoSidedLinOp(_DistributedLinOp):
    """Base class with implementation details for distributed two-sided linear
    operators."""

    def __init__(
        self,
        shape: torch.Size,
        A: List[TwoSidedLinOp],
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
        is_new=True,
        distribution_mode=DistributionMode.ROW,
    ):
        super().__init__(
            shape=shape,
            A=A,
            manager=manager,
            result_queue=result_queue,
            task_queues=task_queues,
            workers=workers,
            is_new=is_new,
            distribution_mode=distribution_mode,
        )

        if self._is_new and not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")

    def _rmatvec(self, w: torch.Tensor):
        if self._distribution_mode == DistributionMode.ROW:
            # Row-distributed operator: chunk by columns, sum results
            results = self._distribute_tasks(
                w, _Operation.RMATVEC, chunk=True, by_dimension=0
            )
            return self._combine_results(results, w, concatenate=False)
        else:  # COLUMN mode
            # Column-distributed operator: send full vector, concatenate results
            results = self._distribute_tasks(w, _Operation.RMATVEC, chunk=False)
            return self._combine_results(results, w, concatenate=True)

    def _rmatmat(self, w: torch.Tensor):
        return self._rmatvec(w)

    def __rmatmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)
            result = self._rmatvec(x.T).T
            return result.squeeze(0)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

    @property
    def T(self):
        # Create a transposed view with shared worker processes
        # When we transpose, we flip the distribution mode
        transposed_mode = (
            DistributionMode.COLUMN
            if self._distribution_mode == DistributionMode.ROW
            else DistributionMode.ROW
        )

        return _DistributedTwoSidedLinOp(
            shape=torch.Size((self.shape[1], self.shape[0])),
            A=[A.T for A in self._A],
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
            is_new=False,
            distribution_mode=transposed_mode,
        )


class _DistributedSymmetricLinOp(_DistributedTwoSidedLinOp):
    """Base class with implementation details for distributed symmetric linear
    operators."""

    def __init__(
        self,
        shape: torch.Size,
        A: List[TwoSidedLinOp],
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
        is_new=True,
        distribution_mode=DistributionMode.ROW,
    ):
        super().__init__(
            shape=shape,
            A=A,
            manager=manager,
            result_queue=result_queue,
            task_queues=task_queues,
            workers=workers,
            is_new=is_new,
            distribution_mode=distribution_mode,
        )

        if self._is_new and shape[0] != shape[1]:
            raise ValueError(
                f"DistributedSymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )

    # Override the _rmatvec and _rmatmat methods since the operator is symmetric
    def _rmatvec(self, w: torch.Tensor):
        return self._matvec(w)

    def _rmatmat(self, w: torch.Tensor):
        return self._matmat(w)

    @property
    def T(self):
        # For symmetric operators, transpose returns self
        return self


# Public classes with simple interfaces
class DistributedLinOp(_DistributedLinOp):
    """Distributed linear operator that performs operations across multiple devices."""

    def __init__(
        self, shape: torch.Size, A: List[LinOp], distribution_mode=DistributionMode.ROW
    ):
        super().__init__(
            shape=shape, A=A, is_new=True, distribution_mode=distribution_mode
        )


class DistributedTwoSidedLinOp(_DistributedTwoSidedLinOp):
    """Distributed two-sided linear operator that performs operations across multiple
    devices."""

    def __init__(
        self,
        shape: torch.Size,
        A: List[TwoSidedLinOp],
        distribution_mode=DistributionMode.ROW,
    ):
        super().__init__(
            shape=shape, A=A, is_new=True, distribution_mode=distribution_mode
        )


class DistributedSymmetricLinOp(_DistributedSymmetricLinOp):
    """Distributed symmetric linear operator that performs operations across multiple
    devices."""

    def __init__(
        self,
        shape: torch.Size,
        A: List[TwoSidedLinOp],
        distribution_mode=DistributionMode.ROW,
    ):
        super().__init__(
            shape=shape, A=A, is_new=True, distribution_mode=distribution_mode
        )
