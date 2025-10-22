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
            B (torch.Tensor): Right-hand side of the linear system. Must be 1D or 2D.
                If 1D with shape (N,), it is automatically resized
                    to 2D with shape (N, 1).
            reg (float): Regularization parameter. Defaults to 0.0.
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

        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("reg", torch.tensor(reg))
        self.w = torch.nn.Parameter(w)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the linear operator (A + reg * I) to tensor v.

        Args:
            v (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of applying the linear operator to v.
        """
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
        return self.A.device

    @property
    def rhs_norm(self) -> torch.Tensor:
        """Get the norm of the right-hand side B.

        Returns:
            torch.Tensor: Norm of B.
        """
        return torch.linalg.norm(self.B, dim=0, ord=2)

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
