"""LinSys module for positive-definite linear systems."""

from warnings import warn

import torch
from linops import LinearOperator, aslinearoperator


class LinSys(torch.nn.Module):
    """Module for regularized linear systems (A + reg * I)w = B."""

    def __init__(
        self,
        A: torch.Tensor | LinearOperator,
        B: torch.Tensor,
        reg: float | torch.Tensor = 0.0,
        w: torch.Tensor | None = None,
    ):
        """Initialize LinSys module.

        Args:
            A (torch.Tensor | LinearOperator): Positive-definite matrix defining
                the linear system.
            B (torch.Tensor): Right-hand side of the linear system. Must be 1D or 2D.
                If 1D with shape (N,), it is automatically resized
                    to 2D with shape (N, 1).
            reg (float | torch.Tensor): Non-negative regularization parameter.
                Defaults to 0.0.
                    Using a tensor allows for differentiation with respect to reg.
            w (torch.Tensor | None): Initial guess for the solution. Defaults to None.
        """
        super().__init__()
        LinSys._check_inputs(A, B, reg, w)

        if w is None:
            w = torch.zeros_like(B)

        # Resize B to 2D for consistent processing
        # When B is resized, we must also resize w accordingly
        if B.ndim == 1:
            B = B.unsqueeze(-1)
            w = w.unsqueeze(-1)

        # Make reg a tensor if it is a float
        if isinstance(reg, float):
            reg = torch.tensor(reg, device=B.device, dtype=B.dtype)

        self.A = aslinearoperator(A)  # Make A a LinearOperator
        self.register_buffer("B", B)
        self.register_buffer("reg", reg)
        self.w = torch.nn.Parameter(w)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the linear operator (A + reg * I) to tensor v.

        Args:
            v (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of applying the linear operator to v.
        """
        if v.ndim == 1:
            warn("Input tensor v is 1D. The input tensor will be unsqueezed to 2D.")
            v = v.unsqueeze(-1)
        return self.A @ v + self.reg * v

    def compute_residual(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the residual of the linear system for a given tensor v.

        Args:
            v (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Residual of the linear system.
        """
        return self.B - self.forward(v)

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
        residual = self.compute_residual(v)
        res_norm = torch.linalg.norm(residual, dim=0, ord=2)
        if relative:
            res_norm /= self.rhs_norm
        return res_norm

    @property
    def device(self) -> torch.device:
        """Get the device of the LinSys module.

        Returns:
            torch.device: Device where the module's tensors are located.
        """
        return self.B.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the LinSys module.

        Returns:
            torch.dtype: Data type of the module's tensors.
        """
        return self.B.dtype

    @property
    def rhs_norm(self) -> torch.Tensor:
        """Get the norm of the right-hand side B.

        Returns:
            torch.Tensor: Norm of B.
        """
        return torch.linalg.norm(self.B, dim=0, ord=2)

    @staticmethod
    def _check_inputs(
        A: torch.Tensor | LinearOperator,
        B: torch.Tensor,
        reg: float | torch.Tensor,
        w: torch.Tensor | None,
    ):
        if not (torch.is_tensor(A) or isinstance(A, LinearOperator)):
            raise TypeError(
                "A must be a torch.Tensor or LinearOperator, "
                f"but received {type(A).__name__}."
            )
        if not torch.is_tensor(B):
            raise TypeError(
                f"B must be a torch.Tensor, but received {type(B).__name__}."
            )
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix or square linear operator.")
        if B.ndim not in [1, 2] or B.shape[0] != A.shape[0]:
            raise ValueError(
                "B must be a tensor whose first dimension matches A's size."
            )

        if not isinstance(reg, (float, torch.Tensor)):
            raise TypeError(
                f"reg must be a float or torch.Tensor, "
                f"but received {type(reg).__name__}."
            )
        if isinstance(reg, float) and reg < 0:
            raise ValueError(f"reg must be a non-negative float, but received {reg}.")
        if isinstance(reg, torch.Tensor):
            if reg.ndim != 0:
                raise ValueError("reg tensor must be a scalar (0-dimensional).")
            if (reg < 0).item():
                raise ValueError(
                    f"reg tensor must be non-negative, but received {reg.item()}."
                )

        if w is not None:
            if not torch.is_tensor(w):
                raise TypeError(
                    f"w must be a torch.Tensor, but received {type(w).__name__}."
                )
            if w.shape != B.shape:
                raise ValueError("w must have the same shape as B.")
