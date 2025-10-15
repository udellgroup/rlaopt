"""LinSys module for positive-definite linear systems."""

import torch


class LinSys(torch.nn.Module):
    """Module for positive-definite linear systems (A + reg * I)w = B."""

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        reg: float = 0.0,
    ):
        """Initialize LinSys module.

        Args:
            A (torch.Tensor): Positive-definite matrix defining the linear system.
            B (torch.Tensor): Right-hand side of the linear system.
            reg (float): Regularization parameter. Defaults to 0.0.
        """
        super().__init__()
        LinSys._check_inputs(A, B, reg)

        self.register_buffer("_A", A)
        self.register_buffer("_B", B)
        self.register_buffer("_reg", torch.tensor(reg))

    @property
    def A(self):
        """Matrix defining the linear system."""
        return self._A

    @property
    def B(self):
        """Right-hand side of the linear system."""
        return self._B

    @property
    def reg(self):
        """Regularization parameter."""
        return self._reg.item()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the linear operator (A + reg * I) to tensor v.

        Args:
            v (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of applying the linear operator to v.
        """
        return self._A @ v + self._reg * v

    @staticmethod
    def _check_inputs(A, B, reg):
        if not torch.is_tensor(A):
            raise TypeError(
                f"A must be a torch.Tensor, but received {type(A).__name__}."
            )
        if not torch.is_tensor(B):
            raise TypeError(
                f"B must be a torch.Tensor, but received {type(B).__name__}."
            )
        if A.ndim != 2 or A.size(0) != A.size(1):
            raise ValueError("A must be a square matrix.")
        if B.ndim not in [1, 2] or B.size(0) != A.size(0):
            raise ValueError(
                "B must be a tensor whose first dimension matches A's size."
            )
        if not isinstance(reg, float) or reg < 0:
            raise ValueError(f"reg must be a non-negative float, but received {reg}.")
