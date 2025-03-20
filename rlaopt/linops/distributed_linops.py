from typing import List

import torch
from torch.multiprocessing import Manager, Process, Queue

from rlaopt.utils import _is_list
from rlaopt.linops.base_linop import BaseLinOp
from rlaopt.linops.linops import LinOp, TwoSidedLinOp

__all__ = ["DistributedLinOp", "DistributedTwoSidedLinOp", "DistributedSymmetricLinOp"]


class DistributedLinOp(BaseLinOp):
    def __init__(
        self,
        shape: torch.Size,
        A: List[LinOp],
    ):
        super().__init__(shape=shape)

        _is_list(A, "A")
        if not all(isinstance(A_i, LinOp) for A_i in A):
            raise ValueError("All elements of A must be linear operators.")
        self._A = A

        # Create device-specific worker processes
        self._manager = Manager()
        self._result_queue = self._manager.Queue()
        self._tasks_queues = {}
        self._workers = {}

        # Start a dedicated worker for each device
        for i, linop in enumerate(A):
            device = linop.device
            if device not in self._tasks_queues:
                self._tasks_queues[device] = Queue()
                worker = Process(
                    target=self._device_worker,
                    args=(device, self._tasks_queues[device], self._result_queue),
                )
                worker.daemon = True
                worker.start()
                self._workers[device] = worker

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

            if operation == "matvec":
                result = linop @ x
            elif operation == "rmatvec":
                result = linop.T @ x
            else:
                raise ValueError(f"Unknown operation: {operation}")

            result_queue.put((task_id, result.cpu()))

    def _matvec(self, w: torch.Tensor):
        # Create a task for each linop and send to the appropriate device queue
        for i, linop in enumerate(self._A):
            self._tasks_queues[linop.device].put((i, linop, w.cpu(), "matvec"))

        # Collect results
        results = [None] * len(self._A)
        for _ in range(len(self._A)):
            task_id, result = self._result_queue.get()
            results[task_id] = result

        # Combine and return
        combined = torch.cat(results, dim=0)
        return combined.to(w.device)

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
        for device, queue in self._tasks_queues.items():
            queue.put(None)  # Signal worker to exit

        for device, worker in self._workers.items():
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()


class DistributedTwoSidedLinOp(DistributedLinOp):
    def __init__(
        self,
        shape: torch.Size,
        A: List[TwoSidedLinOp],
    ):
        super().__init__(shape=shape, A=A)

        if not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")

    def _chunk_vector(self, w: torch.Tensor):
        w_chunks = []
        start_idx = 0
        for i in range(len(self._A)):
            end_idx = start_idx + self._A[i].shape[0]
            w_chunks.append(
                w[start_idx:end_idx].cpu()
            )  # No need to move to device here
            start_idx = end_idx
        return w_chunks

    def _rmatvec(self, w: torch.Tensor):
        w_chunks = self._chunk_vector(w)

        # Dispatch tasks
        for i, (linop, w_chunk) in enumerate(zip(self._A, w_chunks)):
            self._tasks_queues[linop.device].put((i, linop, w_chunk, "rmatvec"))

        # Collect results
        results = [None] * len(self._A)
        for _ in range(len(self._A)):
            task_id, result = self._result_queue.get()
            results[task_id] = result

        # Sum results and return
        result = sum(results)
        return result.to(w.device)

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
        return DistributedTwoSidedLinOp(
            shape=torch.Size((self.shape[1], self.shape[0])),
            A=[A.T for A in self._A],
        )


class DistributedSymmetricLinOp(DistributedTwoSidedLinOp):
    def __init__(self, shape: torch.Size, A: List[TwoSidedLinOp]):
        super().__init__(shape=shape, A=A)

        if shape[0] != shape[1]:
            raise ValueError(
                f"DistributedSymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )

    # Override the _rmatvec and _rmatmat methods since the operator is symmetric
    def _rmatvec(self, w: torch.Tensor):
        return self._matvec(w)

    def _rmatmat(self, w: torch.Tensor):
        return self._matmat(w)
