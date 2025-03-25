from abc import ABC, abstractmethod
from typing import Any, Optional

from pykeops.torch import LazyTensor
import torch

from rlaopt.linops import LinOp, SymmetricLinOp
from rlaopt.utils import _is_torch_tensor


class _KernelLinOp(SymmetricLinOp, ABC):
    def __init__(self, A: torch.Tensor, kernel_params: dict):
        self._check_inputs(A, kernel_params)
        self._A = A
        self._kernel_params = kernel_params
        self._K_lazy = self._get_K()
        super().__init__(
            device=A.device,
            shape=torch.Size((A.shape[0], A.shape[0])),
            matvec=lambda x: self.K_lazy @ x,
            matmat=lambda x: self.K_lazy @ x,
        )

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def kernel_params(self) -> dict:
        return self._kernel_params

    @abstractmethod
    def _check_kernel_params(self, kernel_params: dict):
        pass

    def _check_inputs(self, A: Any, kernel_params: Any):
        _is_torch_tensor(A, "A")
        if A.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got {A.ndim}D tensor.")
        self._check_kernel_params(kernel_params)

    @abstractmethod
    def _kernel_formula(self, Ai_lazy: LazyTensor, Aj_lazy: LazyTensor) -> LazyTensor:
        pass

    def _get_K(
        self, idx1: Optional[torch.Tensor] = None, idx2: Optional[torch.Tensor] = None
    ):
        if idx1 is None:
            Ai_lazy = LazyTensor(self.A[:, None, :])
        else:
            Ai_lazy = LazyTensor(self.A[idx1][:, None, :])

        if idx2 is None:
            Aj_lazy = LazyTensor(self.A[None, :, :])
        else:
            Aj_lazy = LazyTensor(self.A[idx2][None, :, :])

        K_lazy = self._kernel_formula(Ai_lazy, Aj_lazy)
        return K_lazy

    def _get_K_linop(
        self,
        idx1: Optional[torch.Tensor] = None,
        idx2: Optional[torch.Tensor] = None,
        symmetric: bool = False,
    ):
        K = self._get_K(idx1, idx2)
        linop_class = SymmetricLinOp if symmetric else LinOp
        return linop_class(
            device=self.device,
            shape=torch.Size(K.shape),
            matvec=lambda x: K @ x,
            matmat=lambda x: K @ x,
        )

    def row_oracle(self, blk: torch.Tensor):
        _is_torch_tensor(blk, "blk")
        return self._get_K_linop(idx1=blk, symmetric=False)

    def blk_oracle(self, blk: torch.Tensor):
        _is_torch_tensor(blk, "blk")
        return self._get_K_linop(idx1=blk, idx2=blk, symmetric=True)
