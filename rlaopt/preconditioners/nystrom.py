"""This module implements the Nyström preconditioner for optimization algorithms."""

import torch

from .enums import _DampingMode
from .preconditioner import Preconditioner
from .configs import NystromConfig
from rlaopt.sketches import get_sketch


class Nystrom(Preconditioner):
    """Nyström preconditioner class.

    This class implements the Nyström method for approximate matrix inversion,
    which is useful for preconditioning large-scale optimization problems.

    Attributes:
        U (torch.Tensor): Left singular vectors of the approximation.
        S (torch.Tensor): Singular values of the approximation.

    Example:
        >>> import torch
        >>> from rlaopt.preconditioners.configs import NystromConfig
        >>>
        >>> # Create a sample matrix
        >>> A = torch.randn(1000, 1000)
        >>>
        >>> # Configure and initialize the Nyström preconditioner
        >>> config = NystromConfig(rank=100, sketch='gauss', rho=1e-4)
        >>> preconditioner = Nystrom(config)
        >>>
        >>> # Update the preconditioner with the matrix
        >>> preconditioner._update(A, device=torch.device('cpu'))
        >>>
        >>> # Use the preconditioner
        >>> x = torch.randn(1000)
        >>> preconditioned_x = preconditioner @ x
        >>>
        >>> # Use the inverse preconditioner
        >>> inv_preconditioned_x = preconditioner._inv @ x
    """

    def __init__(self, config: NystromConfig):
        """Initialize the Nyström preconditioner.

        Args:
            config (NystromConfig): Configuration for the Nyström preconditioner.
        """
        super().__init__(config)
        self.U = None
        self.S = None
        self.low_precision = False
        self.L = None

    def _update(self, A, device: torch.device) -> None:
        """Update the Nyström preconditioner.

        This method computes the Nyström approximation of the input matrix A.

        Args:
            A (Union[torch.Tensor, LinOpType]): The matrix or linear operator
            for which we are constructing a preconditioner.
            device (torch.device): Device for performing computations.
        """
        # Important signal for the preconditioner inverse
        if A.dtype != torch.float64:
            self.low_precision = True

        # Get sketching matrix
        Omega = get_sketch(
            self.config.sketch,
            "right",
            self.config.rank,
            A.shape[1],
            dtype=A.dtype,
            device=device,
        )
        # Compute sketch
        Y = Omega._apply_right(A)

        # Compute core: Omega.T @ Y
        Core = Omega._apply_left_trans(Y)

        # Get shift for stability
        shift = torch.finfo(Y.dtype).eps * torch.trace(Core)
        Core.diagonal().add_(shift)

        # Compute Nyström approximation
        L = torch.linalg.cholesky(
            Core,
            upper=False,
        )
        # Get eigendecomposition
        B = torch.linalg.solve_triangular(L, Y.T, upper=False)
        self.U, self.S, _ = torch.linalg.svd(B.T, full_matrices=False)
        self.S = torch.maximum(self.S**2 - shift, torch.tensor(0.0))

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        S_safe = self.S if x.ndim == 1 else self.S.unsqueeze(-1)
        return self.U @ (S_safe * (self.U.T @ x)) + self.config.rho * x

    def _inverse_matmul_general(self, x: torch.Tensor, unsqueeze: bool) -> torch.Tensor:
        x_in = x.unsqueeze(-1) if unsqueeze else x
        UTx = self.U.T @ x_in

        # If we are in single precision, we take a more numerically stable approach
        # that requires an additional Cholesky factorization
        if self.low_precision:
            if self.L is None:
                self.L = torch.linalg.cholesky(
                    self.config.rho * torch.diag(self.S**-1) + self.U.T @ self.U
                )
            L_inv_UTx = torch.linalg.solve_triangular(self.L, UTx, upper=False)
            LT_inv_L_inv_UTx = torch.linalg.solve_triangular(
                self.L.T, L_inv_UTx, upper=True
            )
            x_in = 1 / self.config.rho * (x_in - self.U @ LT_inv_L_inv_UTx)
        else:
            x_in = 1 / self.config.rho * (x_in - self.U @ UTx) + self.U @ torch.divide(
                UTx, (self.S + self.config.rho).unsqueeze(-1)
            )
        return x_in.squeeze(-1) if unsqueeze else x_in

    def _inverse_matmul_1d(self, x):
        return self._inverse_matmul_general(x, unsqueeze=True)

    def _inverse_matmul_2d(self, x):
        return self._inverse_matmul_general(x, unsqueeze=False)

    def _update_damping(self, baseline_rho: float) -> None:
        """Update the damping parameter for the Nyström preconditioner.

        This method adjusts the damping parameter based on the damping mode
        specified in the configuration. If the damping mode is adaptive,
        it adds the smallest eigenvalue of the Nyström approximation
        to the baseline damping parameter.

        Args:
            baseline_rho (float): The baseline damping parameter.
        """
        if self.config.damping_mode == _DampingMode.ADAPTIVE:
            self.config.rho = baseline_rho + self.S[-1]
