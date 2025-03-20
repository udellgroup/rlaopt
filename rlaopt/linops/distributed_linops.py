from typing import List

import torch
from torch.multiprocessing import Pool

from rlaopt.utils import _is_list
from rlaopt.linops.base_linop import BaseLinOp
from rlaopt.linops.linops import LinOp, TwoSidedLinOp

__all__ = ["DistributedLinOp", "DistributedTwoSidedLinOp", "DistributedSymmetricLinOp"]


class DistributedLinOp(BaseLinOp):
    def __init__(
        self,
        shape: torch.Size,
        A: List[LinOp],
        pool: Pool,
    ):
        super().__init__(shape=shape)

        _is_list(A, "A")
        if not all(isinstance(A_i, LinOp) for A_i in A):
            raise ValueError("All elements of A must be linear operators.")
        self._A = A
        self._pool = pool

    @staticmethod
    def _worker_matvec(task):
        A_chunk, w_copy = task
        return (A_chunk @ w_copy).cpu()

    def _matvec(self, w: torch.Tensor):
        # Assume the workers' devices are already set correctly during initialization
        tasks = [(A_chunk, w.to(A_chunk.device)) for A_chunk in self._A]
        matvecs = self._pool.map(DistributedLinOp._worker_matvec, tasks)
        result = torch.cat(matvecs, dim=0)
        return result.to(w.device)

    def _matmat(self, w: torch.Tensor):
        return self._matvec(w)

    def __matmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            return self._matvec(x)
        elif x.ndim == 2:
            return self._matmat(x)
        else:
            raise ValueError(f"x must be a 1D or 2D tensor. Received {x.ndim}D tensor.")


class DistributedTwoSidedLinOp(DistributedLinOp):
    def __init__(
        self,
        shape: torch.Size,
        A: List[TwoSidedLinOp],
        pool: Pool,
    ):
        super().__init__(shape=shape, A=A, pool=pool)

        if not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")

    @staticmethod
    def _worker_rmatvec(task):
        A_chunk, w_chunk = task
        return (A_chunk.T @ w_chunk).cpu()

    def _chunk_vector(self, w: torch.Tensor):
        w_chunks = []
        start_idx = 0
        for i in range(len(self._A)):
            end_idx = start_idx + self._A[i].shape[0]
            w_chunks.append(w[start_idx:end_idx].to(self._A[i].device))
            start_idx = end_idx
        return w_chunks

    def _rmatvec(self, w: torch.Tensor):
        w_chunks = self._chunk_vector(w)
        A_chunks = [A_chunk for A_chunk in self._A]
        tasks = list(zip(A_chunks, w_chunks))
        rmatvecs = self._pool.map(DistributedTwoSidedLinOp._worker_rmatvec, tasks)

        result = sum(rmatvecs)
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
            pool=self._pool,
            A=[A.T for A in self._A],
        )


class DistributedSymmetricLinOp(DistributedTwoSidedLinOp):
    def __init__(self, shape: torch.Size, A: List[TwoSidedLinOp], pool: Pool):
        super().__init__(shape=shape, A=A, pool=pool)

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
