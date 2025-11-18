"""Nuclear norm atom for matrix regularization."""

from typing import Any

import torch
from typing_extensions import Self

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


class NucNorm(Atom):
    """Nuclear norm (sum of singular values) of a matrix variable.

    The nuclear norm is defined as the sum of the singular values of a matrix.
    It is commonly used as a convex relaxation of the rank function in low-rank
    matrix optimization problems.

    The atom computes: scaling * ||X||_* = scaling * Σᵢ σᵢ(X)
    where σᵢ(X) are the singular values of X.

    Args:
        x: 2D matrix variable to apply the nuclear norm to.
        scaling: Scaling factor for the nuclear norm. Defaults to 1.0.

    Raises:
        TypeError: If x is not a Variable.
        ValueError: If x is not a 2D matrix.

    Examples:
        >>> X = Variable((10, 5), name='X')
        >>> nuc_norm = NucNorm(X, scaling=0.1)
        >>> loss = nuc_norm.forward()
    """

    def __init__(self, x: Variable, scaling: float | torch.Tensor = 1.0):
        """Initialize the nuclear norm atom.

        Args:
            x: 2D matrix variable to apply the nuclear norm to.
            scaling: Scaling factor for the nuclear norm. Defaults to 1.0.

        Raises:
            TypeError: If x is not a Variable.
            ValueError: If x is not a 2D matrix.
        """
        if x.value.dim() != 2:
            raise ValueError(
                f"Variable value must be 2D Tensor, but got {x.value.dim()}D Tensor."
            )

        super().__init__(
            exprs={"x": x}, buffers={"scaling": scaling}, variable_names=["x"]
        )

    def is_smooth(self) -> bool:
        """Check if the nuclear norm is smooth.

        Returns:
            bool: Always False, as the nuclear norm is non-smooth.
        """
        return False

    def forward(self) -> torch.Tensor:
        """Evaluate the nuclear norm at the registered variable value.

        Returns:
            torch.Tensor: The scaled sum of singular values.
        """
        value = self.get_input("x").forward()
        S = torch.linalg.svdvals(value)
        return self.get_buffer("scaling") * torch.sum(S)

    def is_proxable(self) -> bool:
        """Check if the nuclear norm has a computable proximal operator.

        Returns:
            bool: Always True, as the nuclear norm is proxable.
        """
        return True

    def _prox(
        self, relevant_variable_values: TensorDict, prox_scaling: float
    ) -> TensorDict:
        """Compute the proximal operator of the nuclear norm.

        The proximal operator performs singular value soft-thresholding.
        """
        scale = prox_scaling * self.get_buffer("scaling")

        def prox_func(location: torch.Tensor) -> torch.Tensor:
            if location.shape[0] >= location.shape[1]:
                return _prox_nuc_norm.apply(location, scale)
            else:
                return _prox_nuc_norm.apply(location.T, scale).T

        return relevant_variable_values.apply(prox_func)

    def is_subsamplable(self) -> bool:
        """Check if the nuclear norm supports subsampling.

        Returns:
            bool: Always False, as nuclear norm cannot be subsampled.
        """
        return False

    def subsample(self, indices: torch.Tensor) -> Self:
        """Subsample the nuclear norm (not supported).

        Args:
            indices: Indices to subsample (unused).

        Returns:
            NucNorm: Not applicable.

        Raises:
            NotImplementedError: Nuclear norm cannot be subsampled.
        """
        raise NotImplementedError("Nuclear norm cannot be subsampled")

    def _scale(self, scaling: float) -> Self:
        """Scale the nuclear norm atom."""
        new_scaling = self.get_buffer("scaling") * scaling
        return NucNorm(self.get_input("x"), scaling=new_scaling)


class _prox_nuc_norm(torch.autograd.Function):
    """Proximal operator for the nuclear norm with custom backward pass.

    This implements the proximal operator:
        prox_{λ||·||_*}(X) = argmin_Z (1/2)||Z - X||_F^2 + λ||Z||_*

    where ||·||_* denotes the nuclear norm (sum of singular values) and ||·||_F
    denotes the Frobenius norm.

    The solution is given by soft-thresholding the singular values:
        prox_{λ||·||_*}(X) = U * diag(max(S - λ, 0)) * V^T

    where X = U * diag(S) * V^T is the SVD of X.

    This implementation is adapted from the code accompanying:
    Nobel, Parth, Emmanuel Candès, and Stephen Boyd. "Tractable Evaluation of
    Stein's Unbiased Risk Estimate With Convex Regularizers." IEEE Transactions
    on Signal Processing 71 (2023): 4330-4341.

    Original code: https://github.com/cvxgrp/SURE-CR/blob/main/surecr/prox_lib.py
    (See _prox_nuc_norm class)

    The custom backward pass allows for efficient and numerically stable gradient
    computation through the proximal operator of the nuclear norm.

    Args:
        X (torch.Tensor): Input matrix of shape (m, n) where m >= n
        lambda_ (torch.Tensor): Regularization parameter (threshold for singular values)

    Returns:
        torch.Tensor: Result of applying the proximal operator, shape (m, n)

    Note:
        This implementation assumes m >= n. The full_matrices=True parameter in SVD
        ensures proper handling of the gradient computation in the backward pass.
    """

    @staticmethod
    def forward(ctx: Any, X: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute prox_{λ||·||_*}(X) via SVD and soft-thresholding.

        Args:
            ctx: Context object for saving information needed in backward pass
            X: Input matrix (m, n) with m >= n
            lambda_: Regularization threshold parameter

        Returns:
            Proximal operator applied to X
        """

        def f(s: torch.Tensor) -> torch.Tensor:
            return torch.relu(s - lambda_)

        U, S, VT = torch.linalg.svd(X, full_matrices=True)
        ctx.U = U
        ctx.S = S
        ctx.VT = VT
        ctx.save_for_backward(lambda_)
        return U[:, : X.shape[1]] @ torch.diag(f(S)) @ VT

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass: compute gradient using the method from Nobel et al. (2023).

        This implements the tractable differentiation formula for the nuclear norm
        proximal operator, handling repeated singular values and numerical stability.

        Args:
            ctx: Context object containing saved tensors and SVD components
            grad_output: Gradient of loss with respect to the output

        Returns:
            tuple: (gradient w.r.t. X, None for lambda_)
        """
        epsilon = 1e-3
        (lambda_,) = ctx.saved_tensors
        U, S, VT = ctx.U, ctx.S, ctx.VT
        Z = grad_output
        m, n = Z.shape
        assert m >= n

        def f(s: torch.Tensor) -> torch.Tensor:
            """Soft-thresholding function."""
            return torch.relu(s - lambda_)

        def fp(s: torch.Tensor) -> torch.Tensor:
            """Derivative of soft-thresholding function."""
            return (s >= lambda_).float()

        zeta = U.T @ Z @ VT.T
        Gamma = torch.empty_like(zeta)

        # Handle the case when m > n (extra rows in U)
        if m > n:
            R_mask = S >= epsilon
            R = torch.empty_like(S)
            R[R_mask] = f(S[R_mask]) / S[R_mask]
            R[~R_mask] = fp(S[~R_mask])
            Gamma[n:, :] = torch.tile(R, (m - n, 1)) * zeta[n:, :]

        # Create matrices of repeated singular values for pairwise operations
        S_i = torch.tile(S, (n, 1)).T
        S_j = torch.tile(S, (n, 1))

        def off_diags_zeta(S_i: torch.Tensor, S_j: torch.Tensor) -> torch.Tensor:
            """Weight for off-diagonal elements when singular values differ."""
            return (S_i * f(S_i) - S_j * f(S_j)) / (S_i**2 - S_j**2)

        def off_diags_zetaT(S_i: torch.Tensor, S_j: torch.Tensor) -> torch.Tensor:
            """Weight for transpose contribution when singular values differ."""
            return (S_j * f(S_i) - S_i * f(S_j)) / (S_i**2 - S_j**2)

        zeta_weights = torch.empty_like(zeta[:n, :])
        zetaT_weights = torch.empty_like(zeta[:n, :])

        # Mask for repeated/close singular values (numerical stability)
        mask = torch.abs(S_i - S_j) <= epsilon

        # Fill weights for distinct singular values
        zeta_weights[~mask] = off_diags_zeta(S_i[~mask], S_j[~mask])
        zetaT_weights[~mask] = off_diags_zetaT(S_i[~mask], S_j[~mask])

        def off_diags_pos_repeated_zeta(S: torch.Tensor) -> torch.Tensor:
            """Weight for repeated positive singular values."""
            return 1 / 2 * fp(S) + 1 / 2 * f(S) / S

        def off_diags_pos_repeated_zetaT(S: torch.Tensor) -> torch.Tensor:
            """Transpose weight for repeated positive singular values."""
            return 1 / 2 * fp(S) - 1 / 2 * f(S) / S

        # Handle repeated singular values (above epsilon threshold)
        pos_mask = mask & (S_i >= epsilon)
        zeta_weights[pos_mask] = off_diags_pos_repeated_zeta(S_i[pos_mask])
        zetaT_weights[pos_mask] = off_diags_pos_repeated_zetaT(S_i[pos_mask])

        # Handle repeated near-zero singular values
        zero_mask = mask & (S_i < epsilon)
        zeta_weights[zero_mask] = fp(S_i[zero_mask])
        zetaT_weights[zero_mask] = 0

        # Set diagonal entries
        torch.diagonal(zeta_weights)[:] = fp(S)
        torch.diagonal(zetaT_weights)[:] = 0

        # Compute the top-left block of Gamma
        Gamma[:n, :] = zeta_weights * zeta[:n, :] + zetaT_weights * zeta[:n, :].T

        # Final gradient: rotate back to original coordinate system
        retval = U @ Gamma @ VT
        return retval, None
