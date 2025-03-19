from abc import ABC
from typing import Callable, Optional, Tuple, List

import torch
from torch.multiprocessing import Pool
from torch import vmap


def _check_shape(shape: tuple[int, int]):
    if not isinstance(shape, tuple):
        raise ValueError(f"shape must be a tuple. Received {type(shape)}")
    if len(shape) != 2:
        raise ValueError(f"shape must have two elements. Received {len(shape)}")
    if not all(isinstance(i, int) and i > 0 for i in shape):
        raise ValueError(f"shape must contain positive integers. Received {shape}")


def _check_callable(func: Callable, name: str):
    if not callable(func):
        raise ValueError(f"{name} must be a callable. Received {type(func)}")


class LinOp(ABC):
    def __init__(
        self,
        shape: Tuple[int, int],
        matvec: Callable,
        matmat: Optional[Callable] = None,
    ):
        _check_shape(shape)
        _check_callable(matvec, "matvec")
        if matmat is not None:
            _check_callable(matmat, "matmat")

        self._shape = shape
        self._matvec = matvec

        if matmat is None:
            self._matmat = vmap(self._matvec, in_dims=1, out_dims=1)
        else:
            self._matmat = matmat

    @property
    def shape(self):
        return self._shape

    def __matmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            return self._matvec(x)
        elif x.ndim == 2:
            return self._matmat(x)
        else:
            raise ValueError(f"x must be a 1D or 2D tensor. Received {x.ndim}D tensor.")


class TwoSidedLinOp(LinOp):
    def __init__(
        self,
        shape: Tuple[int, int],
        matvec: Callable,
        rmatvec: Callable,
        matmat: Optional[Callable] = None,
        rmatmat: Optional[Callable] = None,
    ):
        # TODO(pratik): eliminate redundancy in the checks
        _check_shape(shape)
        _check_callable(matvec, "matvec")
        _check_callable(rmatvec, "rmatvec")
        if matmat is not None:
            _check_callable(matmat, "matmat")
        if rmatmat is not None:
            _check_callable(rmatmat, "rmatmat")

        super().__init__(shape, matvec, matmat)

        self._rmatvec = rmatvec
        if rmatmat is None:
            self._rmatmat = vmap(self._rmatvec, in_dims=1, out_dims=1)
        else:
            self._rmatmat = rmatmat

    def __rmatmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(1)
            result = self._rmatvec(x.T).T
            return result.squeeze(1)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

    @property
    def T(self):
        return TwoSidedLinOp(
            shape=(self.shape[1], self.shape[0]),
            matvec=self._rmatvec,
            rmatvec=self._matvec,
        )


class SymmetricLinOp(TwoSidedLinOp):
    def __init__(
        self,
        shape: Tuple[int, int],
        matvec: Callable,
        matmat: Optional[Callable] = None,
    ):
        if shape[0] != shape[1]:
            raise ValueError(
                f"SymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )
        super().__init__(shape, matvec, matvec, matmat, matmat)


# Distributed lin op over rows
class DistributedLinOp(LinOp):
    def __init__(
        self,
        shape: Tuple[int, int],
        A: List[LinOp],
        pool: Pool,
    ):
        if not isinstance(A, list):
            raise ValueError("A must be a list of linear operators.")
        if not all(isinstance(A_i, LinOp) for A_i in A):
            raise ValueError("All elements of A must be linear operators.")

        self._A = A
        self._pool = pool
        super().__init__(shape=shape, matvec=self._matvec, matmat=self._matmat)

    def _worker_matvec(self, task):
        A_chunk, w_copy = task
        return (A_chunk @ w_copy).cpu()

    def _matvec(self, w: torch.Tensor):
        # Assume the workers' devices are already set correctly during initialization
        tasks = [(A_chunk, w.to(A_chunk.device)) for A_chunk in self._A]
        matvecs = self._pool.map(self._worker_matvec, tasks)
        result = torch.cat(matvecs, dim=0)
        return result.to(w.device)

    def _matmat(self, w: torch.Tensor):
        return self._matvec(w)

    def close_pool(self):
        self._pool.close()
        self._pool.join()


class DistributedTwoSidedLinOp(DistributedLinOp):
    def __init__(
        self,
        shape: Tuple[int, int],
        A: List[TwoSidedLinOp],
        pool: Pool,
    ):
        if not isinstance(A, list):
            raise ValueError("A must be a list of linear operators.")
        if not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")

        super().__init__(shape=shape, A=A, pool=pool)

    def _worker_rmatvec(self, task):
        A_chunk, w_chunk = task
        return (A_chunk.T @ w_chunk).cpu()

    def _chunk_vector(self, w: torch.Tensor):
        w_chunks = []
        start_idx = 0
        for i in range(len(self._A)):
            end_idx = start_idx + self._A[i].shape[1]
            w_chunks.append(w[start_idx:end_idx].to(self._A[i].device))
            start_idx = end_idx
        return w_chunks

    def _rmatvec(self, w: torch.Tensor):
        w_chunks = self._chunk_vector(w)
        A_T_chunks = [A_chunk.T for A_chunk in self._A]
        tasks = list(zip(A_T_chunks, w_chunks))
        rmatvecs = self._pool.map(self._worker_rmatvec, tasks)

        result = sum(rmatvecs)
        return result.to(w.device)

    def _rmatmat(self, w: torch.Tensor):
        return self._rmatvec(w)

    def __rmatmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(1)
            result = self._rmatvec(x.T).T
            return result.squeeze(1)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

    @property
    def T(self):
        return DistributedTwoSidedLinOp(
            shape=(self.shape[1], self.shape[0]),
            pool=self._pool,
            A=[A.T for A in self._A],
        )


class DistributedSymmetricLinOp(DistributedTwoSidedLinOp):
    def __init__(
        self,
        shape: Tuple[int, int],
        A: List[TwoSidedLinOp],
        pool: Pool,
    ):
        if shape[0] != shape[1]:
            raise ValueError(
                f"SymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )

        super().__init__(shape=shape, A=A, pool=pool)

        # Override the _rmatvec and _rmatmat methods since the operator is symmetric
        def _rmatvec(self, w: torch.Tensor):
            return self._matvec(w)

        def _rmatmat(self, w: torch.Tensor):
            return self._matmat(w)
