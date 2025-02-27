from abc import ABC, abstractmethod
from rlaopt.utils import TwoSidedLinOp
from rlaopt.utils import _is_str, _is_pos_int

import torch


class Sketch(ABC):
    def __init__(
        self, mode: str, sketch_size: int, matrix_dim: int, device: torch.device
    ):
        self.mode = mode
        self.s = sketch_size
        self.d = matrix_dim
        self.device = device

        _is_str(mode, "mode")
        _is_pos_int(sketch_size, "sketch_size")
        if mode not in ["left", "right"]:
            raise ValueError(f"mode should equal left or right: received {mode}")

        Smat = self._generate_embedding()
        if self.mode == "left":
            shape = (self.s, self.d)
        else:
            shape = (self.d, self.s)

        self.Sop = TwoSidedLinOp(
            shape=shape,
            matvec=lambda v: Smat @ v,
            rmatvec=lambda v: Smat.T @ v,
            matmat=lambda X: Smat @ X,
            rmatmat=lambda X: Smat.T @ X,
        )

    @abstractmethod
    def _generate_embedding(self) -> torch.Tensor:
        pass

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.Sop @ x

    def _apply_transpose(self, x: torch.Tensor) -> torch.Tensor:
        return self.Sop.T @ x
