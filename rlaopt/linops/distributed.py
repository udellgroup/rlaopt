import torch

from .base import _BaseLinOp, _BaseDistributedLinOp
from .enums import _DistributionMode, _Operation


__all__ = [
    "DistributedLinOp",
    "DistributedTwoSidedLinOp",
    "DistributedSymmetricLinOp",
]


class DistributedLinOp(_BaseDistributedLinOp):
    """Distributed linear operator that performs operations across multiple devices."""

    def __init__(self, shape: torch.Size, A: list[_BaseLinOp], distribution_mode: str):
        """Initialize a distributed linear operator.

        Args:
            shape: Shape of the operator as a torch.Size.
            A: List of linear operators to distribute across devices. These can be
               any type of linear operator derived from _BaseLinOp. Each operator
               in the list will handle computation for a portion of the overall
               operation, typically on different devices.
            distribution_mode: Mode for distributing operations.
                "row" - Each operator handles a set of rows of the matrix.
                "column" - Each operator handles a set of columns of the matrix.

        Note:
            The operators in A should be located on different devices (typically GPUs)
            to take advantage of distributed computation. If all operators are on
            the same device, there may be no performance benefit.
        """
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


class DistributedTwoSidedLinOp(DistributedLinOp):
    """Distributed two-sided linear operator that performs operations across multiple
    devices.

    This extends DistributedLinOp to support both left and right matrix multiplication.
    """

    def __init__(
        self,
        shape: torch.Size,
        A: list[_BaseLinOp],
        distribution_mode: str,
    ):
        """Initialize a distributed two-sided linear operator.

        Args:
            shape: Shape of the operator as a torch.Size.
            A: List of linear operators.
               While any _BaseLinOp is accepted in this parameter, the operators
               are expected to properly support right matrix multiplication
               (i.e., the __rmatmul__ method should not raise NotImplementedError).
               Typically these would be TwoSidedLinOp instances or ScaleLinOp instances
               wrapping TwoSidedLinOp.
            distribution_mode: Mode for distributing operations.
                When "row" distribution is used, the transpose will use
                  "column" distribution internally.
                When "column" distribution is used, the transpose will use
                  "row" distribution internally.

        Note:
            If operators in A don't properly support right matrix multiplication,
            operations will fail at runtime when right multiplication is attempted.

        See Also:
            DistributedLinOp: Base class that handles left matrix multiplication only.
        """
        super().__init__(shape=shape, A=A, distribution_mode=distribution_mode)

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
    def T(self) -> "_BaseDistributedLinOp":
        """Return the transpose of the operator.

        Creates a transposed view that shares the same worker processes
          as the original operator.
        The distribution mode is flipped:
        row distribution becomes column distribution and vice versa.

        Returns:
            A distributed linear operator representing the transpose,
              with the same underlying data but
              transposed shape (m,n) → (n,m) and flipped distribution mode.

        Note:
            This operation is lightweight and doesn't duplicate
            the underlying operators, only creating transposed views of them.
        """
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
    devices.

    This extends DistributedTwoSidedLinOp for symmetric operators where left and right
    matrix multiplication operations are identical, and the transpose is the operator
    itself.
    """

    def __init__(
        self,
        shape: torch.Size,
        A: list[_BaseLinOp],
        distribution_mode: str,
    ):
        """Initialize a distributed symmetric linear operator.

        Args:
            shape: Shape of the operator as a torch.Size, must be square (n, n).
            A: List of linear operators. These should be symmetric operators or
               ScaleLinOp instances wrapping symmetric operators.
            distribution_mode: Mode for distributing operations.

        Raises:
            ValueError: If the shape is not square.

        Note:
            For symmetric operators, the distribution mode doesn't change when
            taking the transpose, as the transpose is the operator itself.

        See Also:
            DistributedTwoSidedLinOp: Base class that handles
              two-sided matrix multiplication.
        """
        super().__init__(shape=shape, A=A, distribution_mode=distribution_mode)

        if self._is_new and shape[0] != shape[1]:
            raise ValueError(
                f"DistributedSymmetricLinOp requires the shape to be square. "
                f"The received shape is {shape}."
            )

    # Override the _rmatvec and _rmatmat methods since the operator is symmetric
    def _rmatvec(self, w: torch.Tensor) -> torch.Tensor:
        return self._matvec(w)

    def _rmatmat(self, w: torch.Tensor) -> torch.Tensor:
        return self._matmat(w)

    @property
    def T(self) -> "DistributedSymmetricLinOp":
        """Return the transpose of the operator.

        For symmetric operators, the transpose is the operator itself.

        Returns:
            The operator itself.
        """
        return self
