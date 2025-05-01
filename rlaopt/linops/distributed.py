import torch

from .base import _BaseLinOp, _BaseDistributedLinOp
from .enums import _DistributionMode, _Operation


__all__ = [
    "DistributedLinOp",
    "DistributedTwoSidedLinOp",
    "DistributedSymmetricLinOp",
]


# Private implementation classes with complete functionality
class _DistributedLinOp(_BaseDistributedLinOp):
    """Private implementation of distributed linear operator."""

    def __init__(
        self,
        shape: torch.Size,
        A: list[_BaseLinOp],
        distribution_mode: str,
        is_new: bool = True,
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
    ):
        super().__init__(
            shape=shape,
            A=A,
            distribution_mode=distribution_mode,
            is_new=is_new,
            manager=manager,
            result_queue=result_queue,
            task_queues=task_queues,
            workers=workers,
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


class _DistributedTwoSidedLinOp(_DistributedLinOp):
    """Private implementation of distributed two-sided linear operator."""

    def __init__(
        self,
        shape: torch.Size,
        A: list[_BaseLinOp],
        distribution_mode: str,
        is_new: bool = True,
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
    ):
        super().__init__(
            shape=shape,
            A=A,
            distribution_mode=distribution_mode,
            is_new=is_new,
            manager=manager,
            result_queue=result_queue,
            task_queues=task_queues,
            workers=workers,
        )

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

    @property
    def T(self) -> "_DistributedTwoSidedLinOp":
        """Return the transpose of the distributed two-sided operator."""
        transposed_mode = (
            _DistributionMode.COLUMN
            if self._distribution_mode == _DistributionMode.ROW
            else _DistributionMode.ROW
        )

        # Return a proper _DistributedTwoSidedLinOp, not a base class
        return _DistributedTwoSidedLinOp(
            shape=torch.Size((self.shape[1], self.shape[0])),
            A=[A.T for A in self._A],
            distribution_mode=transposed_mode,
            is_new=False,
            manager=self._manager,
            result_queue=self._result_queue,
            task_queues=self._task_queues,
            workers=self._workers,
        )


class _DistributedSymmetricLinOp(_DistributedTwoSidedLinOp):
    """Private implementation of distributed symmetric linear operator."""

    def __init__(
        self,
        shape: torch.Size,
        A: list[_BaseLinOp],
        distribution_mode: str,
        is_new: bool = True,
        manager=None,
        result_queue=None,
        task_queues=None,
        workers=None,
    ):
        super().__init__(
            shape=shape,
            A=A,
            distribution_mode=distribution_mode,
            is_new=is_new,
            manager=manager,
            result_queue=result_queue,
            task_queues=task_queues,
            workers=workers,
        )

        if is_new and shape[0] != shape[1]:
            raise ValueError(
                f"DistributedSymmetricLinOp requires the shape to be square. "
                f"The received shape is {shape}."
            )

    def _rmatvec(self, w: torch.Tensor) -> torch.Tensor:
        return self._matvec(w)

    def _rmatmat(self, w: torch.Tensor) -> torch.Tensor:
        return self._matmat(w)

    @property
    def T(self) -> "_DistributedSymmetricLinOp":
        """Return the transpose of the distributed symmetric operator (self)."""
        return self


class DistributedLinOp(_DistributedLinOp):
    """Distributed linear operator that performs operations across multiple devices."""

    def __init__(self, shape: torch.Size, A: list[_BaseLinOp], distribution_mode: str):
        """Initialize a distributed linear operator.

        Args:
            shape: Shape of the operator as a torch.Size.
            A: List of linear operators to distribute across devices.
            distribution_mode: Mode for distributing operations ('row' or 'column').
        """
        super().__init__(
            shape=shape, A=A, distribution_mode=distribution_mode, is_new=True
        )


class DistributedTwoSidedLinOp(_DistributedTwoSidedLinOp):
    """Distributed two-sided linear operator that performs operations across multiple
    devices."""

    def __init__(self, shape: torch.Size, A: list[_BaseLinOp], distribution_mode: str):
        """Initialize a distributed two-sided linear operator.

        Args:
            shape: Shape of the operator as a torch.Size.
            A: List of linear operators supporting right matrix multiplication.
            distribution_mode: Mode for distributing operations ('row' or 'column').
        """
        super().__init__(
            shape=shape, A=A, distribution_mode=distribution_mode, is_new=True
        )


class DistributedSymmetricLinOp(_DistributedSymmetricLinOp):
    """Distributed symmetric linear operator that performs operations across multiple
    devices."""

    def __init__(self, shape: torch.Size, A: list[_BaseLinOp], distribution_mode: str):
        """Initialize a distributed symmetric linear operator.

        Args:
            shape: Shape of the operator as a torch.Size, must be square.
            A: List of symmetric linear operators.
            distribution_mode: Mode for distributing operations.
        """
        super().__init__(
            shape=shape, A=A, distribution_mode=distribution_mode, is_new=True
        )
