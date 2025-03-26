"""This module implements the Nyström preconditioner for optimization algorithms."""

import torch

from rlaopt.preconditioners.preconditioner import Preconditioner
from rlaopt.preconditioners.configs import NystromConfig
from rlaopt.preconditioners.sketches.sketch_factory import get_sketch


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
        >>> config = NystromConfig(rank=100, sketch='gaussian', rho=1e-4)
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

    def _update(self, A, device: torch.device) -> None:
        """Update the Nyström preconditioner.

        This method computes the Nyström approximation of the input matrix A.

        Args:
            A (Union[torch.Tensor, LinOpType]): The matrix or linear operator
            for which we are constructing a preconditioner.
            device (torch.device): Device for performing computations.
        """
        # Get sketching matrix
        Omega = get_sketch(
            self.config.sketch, "right", self.config.rank, A.shape[1], device=device
        )
        # Compute sketch
        # Y = A @ Omega
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

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        return self.U @ (self.S * (self.U.T @ x)) + self.config.rho * x

    def _inverse_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with the inverse of the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the inverse matrix multiplication.
        """
        UTx = self.U.T @ x
        x = 1 / self.config.rho * (x - self.U @ UTx) + self.U @ torch.divide(
            UTx, self.S + self.config.rho
        )

        return x

    def _apply_inverse_sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """Applies inverse square root of the preconditioner to a tensor.

           Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the inverse square root matrix multiplication.
        """
        UTx = self.U.T @ x
        x = 1 / (self.config.rho) ** (0.5) * (x - self.U @ UTx) + self.U @ torch.divide(
            UTx, (self.S + self.config.rho) ** (0.5)
        )
        return x
