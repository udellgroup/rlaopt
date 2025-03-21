from abc import ABC, abstractmethod
from typing import Union

from rlaopt.linops import LinOpType
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

        self.Smat = self._generate_embedding()

    @abstractmethod
    def _generate_embedding(self) -> torch.Tensor:
        pass

    def _apply_left(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        return self.Smat @ x

    def _apply_right(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        return x @ self.Smat

    def _apply_left_trans(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        return self.Smat.T @ x

    def _apply_right_trans(self, x: Union[torch.Tensor, LinOpType]) -> torch.Tensor:
        return x @ self.Smat.T
