"""This module implements the Newton preconditioner."""
import torch

from .preconditioner import Preconditioner
from .configs import NewtonConfig


class Newton(Preconditioner):
    """Newton preconditioner class.

    This class implements the Newton method for preconditioning,
    which is useful for accelerating convergence in optimization problems.

    Attributes:
        L (torch.Tensor): Lower triangular Cholesky factor of the preconditioner matrix.

    Example:
        >>> import torch
        >>> from rlaopt.preconditioners.configs import NewtonConfig
        >>>
        >>> # Create a sample matrix
        >>> A = torch.randn(100, 100)
        >>> A = A @ A.T  # Make it positive definite
        >>>
        >>> # Configure and initialize the Newton preconditioner
        >>> config = NewtonConfig(rho=1e-4)
        >>> preconditioner = Newton(config)
        >>>
        >>> # Update the preconditioner with the matrix
        >>> preconditioner._update(A, device=torch.device('cpu'))
        >>>
        >>> # Use the preconditioner
        >>> x = torch.randn(100)
        >>> preconditioned_x = preconditioner @ x
        >>>
        >>> # Use the inverse preconditioner
        >>> inv_preconditioned_x = preconditioner._inv @ x
    """

    def __init__(self, config: NewtonConfig):
        """Initialize the Newton preconditioner.

        Args:
            config (NewtonConfig): Configuration for the Newton preconditioner.
        """
        super().__init__(config)
        self.L = None

    def _update(self, A, device):
        """Update the Newton preconditioner.

        This method computes the Cholesky decomposition of the input matrix A
        with added regularization.

        Args:
            A (Union[torch.Tensor, LinOpType]): The matrix or linear operator
            device (torch.device): The device on which to perform the computations.
        """
        # Handle the tensor and linear operator cases
        if isinstance(A, torch.Tensor):
            A_true = A
        else:
            A_true = A @ torch.eye(A.shape[1], device=device)
        A_true.diagonal().add_(self.config.rho)  # Add rho to the diagonal in-place
        self.L = torch.linalg.cholesky(A_true, upper=False)

    def _matmul(self, x):
        """Perform matrix multiplication with the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        return self.L @ (self.L.T @ x)

    def _inverse_matmul_general(self, x: torch.Tensor, unsqueeze: bool) -> torch.Tensor:
        x_in = x.unsqueeze(-1) if unsqueeze else x
        x_in = torch.linalg.solve_triangular(self.L, x_in, upper=False)
        x_out = torch.linalg.solve_triangular(self.L.T, x_in, upper=True)
        return x_out.squeeze() if unsqueeze else x_out

    def _inverse_matmul_1d(self, x):
        return self._inverse_matmul_general(x, unsqueeze=True)

    def _inverse_matmul_2d(self, x):
        return self._inverse_matmul_general(x, unsqueeze=False)
