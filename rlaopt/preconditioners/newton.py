"""This module implements the Newton preconditioner."""
import torch

from .preconditioner import Preconditioner
from .configs import NewtonConfig
from rlaopt.utils import _is_torch_tensor_1d_2d


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

    def __matmul__(self, x):
        """Perform matrix multiplication with the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        return self.L @ (self.L.T @ x)

    def _inverse_matmul(self, x):
        """Perform matrix multiplication with the inverse of the preconditioner.

        This method solves the system P^(-1)x using forward and backward substitution,
        where P is the preconditioner matrix.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the inverse matrix multiplication.
        """
        _is_torch_tensor_1d_2d(x, "x")
        if x.ndim == 1:
            x = torch.linalg.solve_triangular(self.L, x.unsqueeze(-1), upper=False)
            x = torch.linalg.solve_triangular(self.L.T, x, upper=True)
            return x.squeeze()
        else:
            x = torch.linalg.solve_triangular(self.L, x, upper=False)
            x = torch.linalg.solve_triangular(self.L.T, x, upper=True)
            return x
