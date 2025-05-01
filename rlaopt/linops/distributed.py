import torch

from .base import _BaseDistributedLinOp
from .enums import _DistributionMode, _Operation
from .simple import LinOp, TwoSidedLinOp


__all__ = [
    "DistributedLinOp",
    "DistributedTwoSidedLinOp",
    "DistributedSymmetricLinOp",
]


class DistributedLinOp(_BaseDistributedLinOp):
    """Distributed linear operator that performs operations across multiple devices."""

    def __init__(self, shape: torch.Size, A: list[LinOp], distribution_mode: str):
        super().__init__(
            shape=shape, A=A, distribution_mode=distribution_mode, is_new=True
        )

    def _matvec(self, w: torch.Tensor) -> torch.Tensor:
        if self._distribution_mode == _DistributionMode.ROW:
            # Row-distributed operator: send full vector, concatenate results
            self._distribute_tasks(w, _Operation.MATVEC, chunk=False, by_dimension=0)
            results = self._gather_results(num_tasks=len(self._A))
            return self._combine_results(results, concatenate=True).to(w.device)
        else:  # COLUMN mode
            # Column-distributed operator: chunk by columns, sum results
            self._distribute_tasks(w, _Operation.MATVEC, chunk=True, by_dimension=1)
            results = self._gather_results(num_tasks=len(self._A))
            return self._combine_results(results, concatenate=False).to(w.device)

    def _matmat(self, w: torch.Tensor) -> torch.Tensor:
        return self._matvec(w)

    def __matmul__(self, x):
        if x.ndim == 1:
            return self._matvec(x)
        elif x.ndim == 2:
            return self._matmat(x)
        else:
            raise ValueError(f"x must be a 1D or 2D tensor. Received {x.ndim}D tensor.")


class DistributedTwoSidedLinOp(DistributedLinOp):
    """Distributed two-sided linear operator that performs operations across multiple
    devices."""

    def __init__(
        self,
        shape: torch.Size,
        A: list[TwoSidedLinOp],
        distribution_mode: str,
    ):
        super().__init__(shape=shape, A=A, distribution_mode=distribution_mode)

        if self._is_new and not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")

    def _rmatvec(self, w: torch.Tensor):
        if self._distribution_mode == _DistributionMode.ROW:
            # Row-distributed operator: chunk by columns, sum results
            self._distribute_tasks(w, _Operation.RMATVEC, chunk=True, by_dimension=0)
            results = self._gather_results(num_tasks=len(self._A))
            return self._combine_results(results, concatenate=False).to(w.device)
        else:  # COLUMN mode
            # Column-distributed operator: send full vector, concatenate results
            self._distribute_tasks(w, _Operation.RMATVEC, chunk=False, by_dimension=1)
            results = self._gather_results(num_tasks=len(self._A))
            return self._combine_results(results, concatenate=True).to(w.device)

    def _rmatmat(self, w: torch.Tensor):
        return self._rmatvec(w)

    def __rmatmul__(self, x):
        if x.ndim == 1:
            return self._rmatvec(x)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

    @property
    def T(self) -> "_BaseDistributedLinOp":
        # Create a transposed view with shared worker processes
        # When we transpose, we flip the distribution mode
        transposed_mode = (
            _DistributionMode.COLUMN
            if self._distribution_mode == _DistributionMode.ROW
            else _DistributionMode.ROW
        )

        return _BaseDistributedLinOp(
            shape=torch.Size((self.shape[1], self.shape[0])),
            A=[A.T for A in self._A],
            distribution_mode=transposed_mode,
            is_new=False,
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
        )


class DistributedSymmetricLinOp(DistributedTwoSidedLinOp):
    """Distributed symmetric linear operator that performs operations across multiple
    devices."""

    def __init__(
        self,
        shape: torch.Size,
        A: list[TwoSidedLinOp],
        distribution_mode: str,
    ):
        super().__init__(shape=shape, A=A, distribution_mode=distribution_mode)

        if self._is_new and shape[0] != shape[1]:
            raise ValueError(
                f"DistributedSymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )

    # Override the _rmatvec and _rmatmat methods since the operator is symmetric
    def _rmatvec(self, w: torch.Tensor) -> torch.Tensor:
        return self._matvec(w)

    def _rmatmat(self, w: torch.Tensor) -> torch.Tensor:
        return self._matmat(w)

    @property
    def T(self) -> "DistributedSymmetricLinOp":
        # For symmetric operators, transpose returns self
        return self
