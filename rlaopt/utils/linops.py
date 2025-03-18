from abc import ABC
from typing import Callable, Optional, Union, Tuple, List

import torch
import torch.multiprocessing as mp
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
        device_ids: List[Union[int, torch.device]],
        A: List[LinOp],
    ):
        if not isinstance(device_ids, list):
            raise ValueError("device_ids must be a list of device IDs.")
        if not all(
            isinstance(device_id, (int, torch.device)) for device_id in device_ids
        ):
            raise ValueError("device_ids must contain integers or torch devices.")
        if not isinstance(A, list):
            raise ValueError("A must be a list of linear operators.")
        if not all(isinstance(A_i, LinOp) for A_i in A):
            raise ValueError("All elements of A must be linear operators.")
        if len(device_ids) != len(A):
            raise ValueError("device_ids and A must have the same length.")

        self._A = A
        self._device_ids = self._init_device_ids(device_ids)

        # initialize workers
        self._pool = mp.Pool(processes=len(device_ids))

        shape = (sum(A_i.shape[0] for A_i in A), A[0].shape[1])
        super().__init__(shape=shape, matvec=self._matvec)

    def _init_device_ids(self, device_ids: List[Union[int, torch.device]]):
        device_ids_corrected = []
        for device_id in device_ids:
            if isinstance(device_id, int):
                device_ids_corrected.append(torch.device(f"cuda:{device_id}"))
            elif isinstance(device_id, torch.device):
                device_ids_corrected.append(device_id)

        return device_ids_corrected

    def _worker(self, task):
        A_chunk, w_copy = task
        return (A_chunk @ w_copy).cpu()

    def _matvec(self, w: torch.Tensor):
        # Send w to each device
        w_copies = [w.to(device_id) for device_id in self._device_ids]

        # Compute matvecs in parallel
        tasks = list(zip(self._A, w_copies))
        matvecs = self._pool.map(self._worker, tasks)
        result = torch.cat(matvecs, dim=0)
        return result.to(w.device)


class DistributedTwoSidedLinOp(TwoSidedLinOp):
    def __init__(
        self,
        device_ids: List[Union[int, torch.device]],
        A: List[TwoSidedLinOp],
    ):
        if not isinstance(device_ids, list):
            raise ValueError("device_ids must be a list of device IDs.")
        if not all(
            isinstance(device_id, (int, torch.device)) for device_id in device_ids
        ):
            raise ValueError("device_ids must contain integers or torch devices.")
        if not isinstance(A, list):
            raise ValueError("A must be a list of linear operators.")
        if not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")
        if len(device_ids) != len(A):
            raise ValueError("device_ids and A must have the same length.")

        # Initialize members specific to distributed processing
        self._A = A
        self._device_ids = self._init_device_ids(device_ids)
        self._pool = mp.Pool(processes=len(device_ids))

        shape = (sum(A_i.shape[0] for A_i in A), A[0].shape[1])
        super().__init__(
            shape=shape,
            matvec=self._matvec,
            rmatvec=self._rmatvec,
            matmat=None,
            rmatmat=None,
        )

    def _init_device_ids(self, device_ids: List[Union[int, torch.device]]):
        device_ids_corrected = []
        for device_id in device_ids:
            if isinstance(device_id, int):
                device_ids_corrected.append(torch.device(f"cuda:{device_id}"))
            elif isinstance(device_id, torch.device):
                device_ids_corrected.append(device_id)
        return device_ids_corrected

    def _worker_matvec(self, task):
        A_chunk, w_copy = task
        return (A_chunk @ w_copy).cpu()

    def _worker_rmatvec(self, task):
        A_chunk, w_chunk = task
        return (A_chunk.T @ w_chunk).cpu()

    def _matvec(self, w: torch.Tensor):
        w_copies = [w.to(device_id) for device_id in self._device_ids]
        tasks = list(zip(self._A, w_copies))
        matvecs = self._pool.map(self._worker_matvec, tasks)
        result = torch.cat(matvecs, dim=0)
        return result.to(w.device)

    def _rmatvec(self, w: torch.Tensor):
        # Split w based on A chunks
        w_chunks = []
        start_idx = 0
        for i in range(len(self._device_ids)):
            end_idx = start_idx + self._A[i].shape[1]
            w_chunks.append(w[start_idx:end_idx].to(self._device_ids[i]))
            start_idx = end_idx

        A_T_chunks = [A_chunk.T for A_chunk in self._A]
        tasks = list(zip(A_T_chunks, w_chunks))
        rmatvecs = self._pool.map(self._worker_rmatvec, tasks)

        # Since the result will be summed over all chunks and added to produce
        # the final output vector
        result = sum(rmatvecs)
        return result.to(w.device)


class DistributedSymmetricLinOp(DistributedTwoSidedLinOp):
    def __init__(
        self,
        device_ids: List[Union[int, torch.device]],
        A: List[TwoSidedLinOp],
    ):
        if not isinstance(device_ids, list):
            raise ValueError("device_ids must be a list of device IDs.")
        if not all(
            isinstance(device_id, (int, torch.device)) for device_id in device_ids
        ):
            raise ValueError("device_ids must contain integers or torch devices.")
        if not isinstance(A, list):
            raise ValueError("A must be a list of linear operators.")
        if not all(isinstance(A_i, TwoSidedLinOp) for A_i in A):
            raise ValueError("All elements of A must be two-sided linear operators.")
        if len(device_ids) != len(A):
            raise ValueError("device_ids and A must have the same length.")
        if not all(A_i.shape[0] == A_i.shape[1] for A_i in A):
            raise ValueError("All elements of A must be square matrices.")

        # Initialize members specific to distributed processing
        self._A = A
        self._device_ids = self._init_device_ids(device_ids)
        self._pool = mp.Pool(processes=len(device_ids))

        shape = (sum(A_i.shape[0] for A_i in A), A[0].shape[1])
        super().__init__(
            shape=shape,
            matvec=self._matvec,
            rmatvec=self._matvec,
            matmat=None,
            rmatmat=None,
        )

    def _init_device_ids(self, device_ids: List[Union[int, torch.device]]):
        device_ids_corrected = []
        for device_id in device_ids:
            if isinstance(device_id, int):
                device_ids_corrected.append(torch.device(f"cuda:{device_id}"))
            elif isinstance(device_id, torch.device):
                device_ids_corrected.append(device_id)
        return device_ids_corrected

    def _worker_matvec(self, task):
        A_chunk, w_copy = task
        return (A_chunk @ w_copy).cpu()

    def _matvec(self, w: torch.Tensor):
        w_copies = [w.to(device_id) for device_id in self._device_ids]
        tasks = list(zip(self._A, w_copies))
        matvecs = self._pool.map(self._worker_matvec, tasks)
        result = torch.cat(matvecs, dim=0)
        return result.to(w.device)
