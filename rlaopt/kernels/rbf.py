from functools import partial
from typing import Set

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import (
    LinOp,
    TwoSidedLinOp,
    SymmetricLinOp,
    DistributionMode,
    DistributedSymmetricLinOp,
)

__all__ = ["RBFLinOp", "DistributedRBFLinOpV1"]


def _matvec(x: torch.Tensor, K: LazyTensor):
    return K @ x


def _Kb_matvec(x: torch.Tensor, A: torch.Tensor, chunk_idx: torch.Tensor, sigma: float):
    # Compute the kernel matrix
    Ab_lazy = LazyTensor(A[chunk_idx][:, None, :])
    A_lazy = LazyTensor(A[None, :, :])
    D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
    Kb = (-D / (2 * sigma**2)).exp()
    return Kb @ x


def _Kb_rmatvec(
    x: torch.Tensor, A: torch.Tensor, chunk_idx: torch.Tensor, sigma: float
):
    # Compute the kernel matrix
    Ab_lazy = LazyTensor(A[chunk_idx][None, :, :])
    A_lazy = LazyTensor(A[:, None, :])
    D = ((A_lazy - Ab_lazy) ** 2).sum(dim=2)
    KbT = (-D / (2 * sigma**2)).exp()
    return KbT @ x


class RBFLinOp(SymmetricLinOp):
    def __init__(self, A: torch.Tensor, sigma: float):
        """Initialize the RBF kernel.

        Args:
            A (torch.Tensor): The input data.
            sigma (float): The bandwidth parameter for the RBF kernel.
        """
        self.A = A
        self.sigma = sigma
        K = self._get_K()
        super().__init__(
            device=A.device,
            shape=torch.Size((A.shape[0], A.shape[0])),
            matvec=partial(_matvec, K=K),
            matmat=partial(_matvec, K=K),
        )

    def _get_K(
        self,
        idx1: torch.Tensor = None,
        idx2: torch.Tensor = None,
    ):
        if idx1 is None:
            Ai_lazy = LazyTensor(self.A[:, None, :])
        else:
            Ai_lazy = LazyTensor(self.A[idx1][:, None, :])

        if idx2 is None:
            Aj_lazy = LazyTensor(self.A[None, :, :])
        else:
            Aj_lazy = LazyTensor(self.A[idx2][None, :, :])

        D = ((Ai_lazy - Aj_lazy) ** 2).sum(dim=2)
        K = (-D / (2 * self.sigma**2)).exp()
        return K

    def _get_K_linop(
        self,
        idx1: torch.Tensor = None,
        idx2: torch.Tensor = None,
        symmetric: bool = False,
    ):
        K = self._get_K(idx1=idx1, idx2=idx2)
        linop_class = SymmetricLinOp if symmetric else LinOp
        K_linop = linop_class(
            device=self.A.device,
            shape=torch.Size(K.shape),
            matvec=lambda x: K @ x,
            matmat=lambda x: K @ x,
        )
        return K_linop

    def row_oracle(self, blk: torch.Tensor):
        return self._get_K_linop(idx1=blk, symmetric=False)

    def blk_oracle(self, blk: torch.Tensor):
        return self._get_K_linop(idx1=blk, idx2=blk, symmetric=True)


# For PCG only
class DistributedRBFLinOpV1(DistributedSymmetricLinOp):
    def __init__(self, A: torch.Tensor, sigma: float, devices: Set[torch.device]):
        A_chunk_idx = torch.chunk(torch.arange(A.shape[0]), len(devices), dim=0)

        lin_ops = []
        for device, chunk_idx in zip(devices, A_chunk_idx):
            matvec_fn = partial(
                _Kb_matvec, A=A.to(device), chunk_idx=chunk_idx, sigma=sigma
            )
            rmatvec_fn = partial(
                _Kb_rmatvec, A=A.to(device), chunk_idx=chunk_idx, sigma=sigma
            )
            lin_ops.append(
                TwoSidedLinOp(
                    device=device,
                    shape=torch.Size((chunk_idx.shape[0], A.shape[0])),
                    matvec=matvec_fn,
                    rmatvec=rmatvec_fn,
                    matmat=matvec_fn,
                    rmatmat=rmatvec_fn,
                )
            )

        super().__init__(
            shape=torch.Size((A.shape[0], A.shape[0])),
            A=lin_ops,
            distribution_mode=DistributionMode.ROW,
        )
