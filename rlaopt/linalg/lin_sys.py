"""LinSys module for positive-definite linear systems."""

import torch


class LinSys(torch.nn.Module):
    """Module for positive-definite linear systems (A + reg * I)w = B."""

    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        reg: float = 0.0,
        w: torch.Tensor | None = None,
    ):
        """Initialize LinSys module.

        Args:
            A (torch.Tensor): Positive-definite matrix defining the linear system.
            B (torch.Tensor): Right-hand side of the linear system.
            reg (float): Regularization parameter. Defaults to 0.0.
            w (torch.Tensor | None): Initial guess for the solution. Defaults to None.
        """
        super().__init__()
        LinSys._check_inputs(A, B, reg, w)

        if w is None:
            w = torch.zeros_like(B)

        self.register_buffer("_A", A)
        self.register_buffer("_B", B)
        self.register_buffer("_reg", torch.tensor(reg))
        self.w = torch.nn.Parameter(w)

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

    def compute_residual_norm(
        self, v: torch.Tensor, relative: bool = False
    ) -> torch.Tensor:
        """Compute the residual norm of the linear system for a given tensor v.

        Args:
            v (torch.Tensor): Input tensor.
            relative (bool): If True, return the relative residual norm.
                Defaults to False.

        Returns:
            torch.Tensor: Residual norm of the linear system.
        """
        B = self._B

        residual = self.forward(v) - B
        res_norm = torch.norm(residual, dim=0, ord=2)
        if relative:
            b_norm = torch.norm(B, dim=0, ord=2)
            res_norm /= b_norm
        return res_norm

    @staticmethod
    def _check_inputs(A, B, reg, w):
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

        if w is not None:
            if not torch.is_tensor(w):
                raise TypeError(
                    f"w must be a torch.Tensor, but received {type(w).__name__}."
                )
            if w.shape != B.shape:
                raise ValueError("w must have the same shape as B.")
