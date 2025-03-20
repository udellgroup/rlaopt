from abc import ABC
from typing import Any, Callable, Optional, List

import torch
from torch.multiprocessing import Pool
from torch import vmap

from rlaopt.utils.input_checkers import (
    _is_callable,
    _is_list,
    _is_torch_device,
    _is_torch_size,
)


def _is_linop_or_torch_tensor(param: Any, param_name: str):
    if not isinstance(param, (LinOp, DistributedLinOp, torch.Tensor)):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type LinOp, DistributedLinOp, or torch.Tensor"
        )


def _check_shape(shape: Any):
    _is_torch_size(shape, "shape")
    if len(shape) != 2:
        raise ValueError(f"shape must have two elements. Received {len(shape)}")
    if not all(isinstance(i, int) and i > 0 for i in shape):
        raise ValueError(f"shape must contain positive integers. Received {shape}")


class LinOp(ABC):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        matmat: Optional[Callable] = None,
    ):
        _is_torch_device(device, "device")
        _check_shape(shape)
        _is_callable(matvec, "matvec")
        if matmat is not None:
            _is_callable(matmat, "matmat")

        self._device = device
        self._shape = shape
        self._matvec = matvec

        if matmat is None:
            self._matmat = vmap(self._matvec, in_dims=1, out_dims=1)
        else:
            self._matmat = matmat

    @property
    def device(self):
        return self._device

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
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        rmatvec: Callable,
        matmat: Optional[Callable] = None,
        rmatmat: Optional[Callable] = None,
    ):
        _is_callable(rmatvec, "rmatvec")
        if rmatmat is not None:
            _is_callable(rmatmat, "rmatmat")

        super().__init__(device, shape, matvec, matmat)

        self._rmatvec = rmatvec
        if rmatmat is None:
            self._rmatmat = vmap(self._rmatvec, in_dims=1, out_dims=1)
        else:
            self._rmatmat = rmatmat

    def __rmatmul__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(0)
            result = self._rmatvec(x.T).T
            return result.squeeze(0)
        elif x.ndim == 2:
            return self._rmatmat(x.T).T

    @property
    def T(self):
        return TwoSidedLinOp(
            device=self.device,
            shape=torch.Size((self.shape[1], self.shape[0])),
            matvec=self._rmatvec,
            rmatvec=self._matvec,
            matmat=self._rmatmat,
            rmatmat=self._matmat,
        )


class SymmetricLinOp(TwoSidedLinOp):
    def __init__(
        self,
        device: torch.device,
        shape: torch.Size,
        matvec: Callable,
        matmat: Optional[Callable] = None,
    ):
        if shape[0] != shape[1]:
            raise ValueError(
                f"SymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )
        super().__init__(device, shape, matvec, matvec, matmat, matmat)


# Distributed lin op over rows
class DistributedLinOp:
    def __init__(
        self,
        shape: torch.Size,
        A: List[LinOp],
        pool: Pool,
    ):
        _is_torch_size(shape, "shape")
        _is_list(A, "A")
        if not all(isinstance(A_i, LinOp) for A_i in A):
            raise ValueError("All elements of A must be linear operators.")

        self._shape = shape
        self._A = A
        self._pool = pool

    @property
    def shape(self):
        return self._shape

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
        _is_torch_size(shape, "shape")
        _is_list(A, "A")
        if not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")

        super().__init__(shape=shape, A=A, pool=pool)

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
        if shape[0] != shape[1]:
            raise ValueError(
                f"DistributedSymmetricLinOp requires the shape to be square. \
                    The received shape is {shape}."
            )

        super().__init__(shape=shape, A=A, pool=pool)

    # Override the _rmatvec and _rmatmat methods since the operator is symmetric
    def _rmatvec(self, w: torch.Tensor):
        return self._matvec(w)

    def _rmatmat(self, w: torch.Tensor):
        return self._matmat(w)
