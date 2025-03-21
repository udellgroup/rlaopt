from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import LinOp, SymmetricLinOp


def matvec(x: torch.Tensor, K: LazyTensor):
    return K @ x


class RBF:
    def __init__(self, A: torch.Tensor, sigma: float):
        """Initialize the RBF kernel.

        Args:
            A (torch.Tensor): The input data.
            sigma (float): The bandwidth parameter for the RBF kernel.
        """
        self.A = A
        self.sigma = sigma
        self.K_linop = self._get_K_linop(symmetric=True)

    def _get_K_linop(
        self,
        idx1: torch.Tensor = None,
        idx2: torch.Tensor = None,
        symmetric: bool = False,
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

    def __matmul__(self, x: torch.Tensor):
        return self.K_linop @ x
