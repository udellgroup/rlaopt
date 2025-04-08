"""This module implements the Sketch and Precondition Preconditioner."""

import gc
import torch

from .preconditioner import Preconditioner
from .configs import SkPreConfig
from rlaopt.sketches import get_sketch


class SkPre(Preconditioner):
    """Sketched Preconditioner (SkPre) class.

    This class implements a preconditioner based on matrix sketching techniques,
    which is useful for preconditioning large-scale optimization problems.

    Attributes:
        Y (torch.Tensor): The sketched matrix.
        L (torch.Tensor): The Cholesky factor used in preconditioning.

    Example:
        >>> import torch
        >>> from rlaopt.preconditioners.configs import SkPreConfig
        >>>
        >>> # Create a sample matrix
        >>> A = torch.randn(1000, 500)
        >>>
        >>> # Configure and initialize the SkPre preconditioner
        >>> config = SkPreConfig(sketch='gauss', sketch_size=200, rho=1e-4)
        >>> preconditioner = SkPre(config)
        >>>
        >>> # Update the preconditioner with the matrix
        >>> preconditioner._update(A, device=torch.device('cpu'))
        >>>
        >>> # Use the preconditioner
        >>> x = torch.randn(500)
        >>> preconditioned_x = preconditioner @ x
        >>>
        >>> # Use the inverse preconditioner
        >>> inv_preconditioned_x = preconditioner._inv @ x
    """

    def __init__(self, config: SkPreConfig):
        """Initialize the Sketched Preconditioner.

        Args:
            config (SkPreConfig): Configuration for the Sketched Preconditioner.
        """
        super().__init__(config)
        self.Y = None
        self.L = None

    def _update(self, A, device):
        """Update the Sketched Preconditioner.

        This method computes the sketched matrix and constructs the preconditioner.

        Args:
            A (Union[torch.Tensor, LinOpType]): The matrix or linear operator
            for which we are constructing a preconditioner.
            device (torch.device): The device on which to perform the computations.
        """
        # Get sketching matrix
        Omega = get_sketch(
            self.config.sketch,
            "left",
            self.config.sketch_size,
            A.shape[0],
            device=device,
        )

        # Compute sketch: Y = Omega @ A
        self.Y = Omega._apply_left(A)

        # Construct preconditioner
        if self.config.rho == 0:
            # No damping
            # Get L from Cholesky of Y.T @ Y

            # Compute gram matrix
            G = self.Y.T @ self.Y
            # Delete Y as it won't be needed
            self._del_Y()
            # Get lower Cholesky factor of G
            self.L = torch.linalg.cholesky(G, upper=False)

        else:
            # Damping
            # Two cases based on sketch size
            # Case 1: sketch_size >= number columns of A
            if self.config.sketch_size >= A.shape[1]:
                # Get Gram matrix
                G = self.Y.T @ self.Y
                # Delete Y as it won't be needed
                self._del_Y()
                # Add damping
                G.diagonal().add_(self.config.rho)
                # Get lower Cholesky factor of G
                self.L = torch.linalg.cholesky(G, upper=False)

            else:
                # Case 2: sketch_size<= number of columns of A
                # Get lower Cholesky L of Y @ Y.T
                # L will then be used with Woodbury matrix identity

                # Get Gram matrix
                G = self.Y @ self.Y.T
                # Add damping
                G.diagonal().add_(self.config.rho)
                # Get lower Cholesky factor of G
                self.L = torch.linalg.cholesky(G, upper=False)

    def _matmul(self, x):
        """Perform matrix multiplication with the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        if self.Y is None:
            return self.L @ (self.L.T @ x)
        else:
            return self.Y.T @ (self.Y @ x) + self.config.rho * x

    def _inverse_matmul_Y_none(self, x: torch.Tensor, unsqueeze: bool) -> torch.Tensor:
        # Computation done by two triangular solves
        x_in = x.unsqueeze(-1) if unsqueeze else x
        L_inv_x = torch.linalg.solve_triangular(self.L.T, x_in, upper=True)
        P_inv_x = torch.linalg.solve_triangular(self.L, L_inv_x, upper=False)
        return P_inv_x.squeeze() if unsqueeze else P_inv_x

    def _inverse_matmul_Y_exists(
        self, x: torch.Tensor, unsqueeze: bool
    ) -> torch.Tensor:
        # Computaiton uses Woodbury identity
        x_in = x.unsqueeze(-1) if unsqueeze else x
        Yx = self.Y @ x_in
        L_inv_Yx = torch.linalg.solve_triangular(self.L, Yx, upper=False)
        LT_inv_L_inv_Yx = torch.linalg.solve_triangular(self.L.T, L_inv_Yx, upper=True)
        P_inv_x = 1 / self.config.rho * (x_in - self.Y.T @ LT_inv_L_inv_Yx)
        return P_inv_x.squeeze() if unsqueeze else P_inv_x

    def _inverse_matmul_1d(self, x):
        """Perform matrix multiplication with the inverse of the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with. Assumed to be 1D.

        Returns:
            torch.Tensor: The result of the inverse matrix multiplication.
        """
        if self.Y is None:
            return self._inverse_matmul_Y_none(x, unsqueeze=True)
        else:
            return self._inverse_matmul_Y_exists(x, unsqueeze=True)

    def _inverse_matmul_2d(self, x):
        """Perform matrix multiplication with the inverse of the preconditioner.

        Args:
            x (torch.Tensor): The tensor to multiply with. Assumed to be 2D.

        Returns:
            torch.Tensor: The result of the inverse matrix multiplication.
        """
        if self.Y is None:
            return self._inverse_matmul_Y_none(x, unsqueeze=False)
        else:
            return self._inverse_matmul_Y_exists(x, unsqueeze=False)

    def _del_Y(self):
        """Delete the sketched matrix Y and free up memory.

        This method is called when Y is no longer needed to save memory, especially
        important for GPU computations.
        """
        device = self.Y.device
        self.Y = None
        # Free up memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
