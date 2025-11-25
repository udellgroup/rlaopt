"""Nyström preconditioner and configuration."""

from typing import Literal
from warnings import warn

import torch
from linops import LinearOperator
from pydantic import Field, model_validator
from typing_extensions import Self

from rlaopt.linalg.preconditioners.preconditioner import (
    Preconditioner,
    PreconditionerConfig,
)


class NystromConfig(PreconditionerConfig):
    """Configuration for the Nyström preconditioner.

    Attributes:
        rank_init: Initial rank of the Nyström approximation.
        rank_max: Maximum allowable rank.
        num_power_iters: Number of power iterations for error estimation
            in rank adaptation.
        error_tolerance: Error tolerance for rank adaptation.
        base_damping: Base damping parameter.
        damping_mode: Damping mode, either 'adaptive' or 'non_adaptive'.
        dtype: Data type for computations.
    """

    # TODO(pratik): add option for sketching method

    rank_init: int = Field(
        gt=0, description="Initial rank of the Nyström approximation."
    )
    rank_max: int | None = Field(default=None, description="Maximum allowable rank.")
    num_power_iters: int = Field(
        default=10,
        gt=0,
        description="Number of power iterations for error estimation"
        " in rank adaptation.",
    )
    error_tolerance: float = Field(
        default=1e-2, gt=0.0, description="Error tolerance for rank adaptation."
    )
    base_damping: float = Field(ge=0.0, description="Base damping parameter.")
    damping_mode: Literal["adaptive", "non_adaptive"] = Field(
        default="adaptive",
        description="Damping mode: 'adaptive' adjusts based on smallest eigenvalue,"
        " 'non_adaptive' uses base_damping only.",
    )

    @model_validator(mode="after")
    def validate_rank_max(self) -> Self:
        """Validate and set rank_max based on rank_init.

        If rank_max is None, it is set to rank_init.
        If rank_max is provided, it must be >= rank_init.
        """
        if self.rank_max is None:
            self.rank_max = self.rank_init
        elif self.rank_max < self.rank_init:
            raise ValueError(
                f"rank_max ({self.rank_max}) must be >= rank_init ({self.rank_init})"
            )
        return self


class Nystrom(Preconditioner):
    """Nyström preconditioner implementation with adaptive rank."""

    def __init__(self, config: NystromConfig):
        """Initialize the Nyström preconditioner with the given configuration.

        Args:
            config (NystromConfig): Configuration for the Nyström preconditioner.
        """
        super().__init__(config)
        self.U = None
        self.S = None
        self.L = None
        self.current_damping = None
        self.using_low_precision = False

    def _update(self, A: torch.Tensor | LinearOperator, dtype: torch.dtype):
        """Update the Nyström preconditioner based on the matrix A."""
        # Unpack config
        num_cols_to_add = self._config.rank_init
        error_tolerance = self._config.error_tolerance
        num_power_iters = self._config.num_power_iters
        rank_max = self._config.rank_max
        damping_mode = self._config.damping_mode
        base_damping = self._config.base_damping

        if dtype != torch.float64:
            self.using_low_precision = True

        device = A.device
        n = A.shape[0]

        # Initialize sketching matrix and sketch
        Omega = torch.empty((n, 0), dtype=dtype, device=device)
        Y = Omega.clone()

        # Initialize empty tensors for estimated eigenvectors and eigenvalues
        U = torch.Tensor([])
        S = torch.Tensor([])

        # Start error at infinity to enter the loop
        error = torch.inf
        break_early = False

        # Compute Nyström approximation with adaptive rank
        while error > error_tolerance:
            # Update sketching matrix and sketch
            Omega_new = _generate_ortho_embedding(
                dimension=n,
                sketch_size=num_cols_to_add,
                dtype=dtype,
                device=device,
            )
            Y_new = A @ Omega_new
            Omega = torch.hstack([Omega, Omega_new])
            Y = torch.hstack([Y, Y_new])

            # Compute core
            Core = Omega.T @ Y

            # Shift for stability
            shift = torch.finfo(Y.dtype).eps * torch.trace(Core)
            Core.diagonal().add_(shift)

            L = torch.linalg.cholesky(Core, upper=False)

            # Get eigendecomposition
            B = torch.linalg.solve_triangular(L, Y.T, upper=False)
            U, sigma, _ = torch.linalg.svd(B.T, full_matrices=False)
            S = torch.nn.functional.relu(sigma**2 - shift)

            # Make sure to estimate error before possibly breaking early
            error = _randomized_power_err_est(A, U, S, num_power_iters, dtype)

            if break_early:
                break

            # Increase sketch size for next iteration
            num_cols_to_add = Omega.shape[1]
            if 2 * num_cols_to_add > rank_max:
                num_cols_to_add = rank_max - Omega.shape[1]
                break_early = True

        if error > error_tolerance:
            warn(
                f"Reached maximum rank {rank_max} before achieving "
                f"desired error tolerance {error_tolerance}."
            )

        self.U = U
        self.S = S

        # Recalculate damping
        if damping_mode == "adaptive":
            self.current_damping = base_damping + self.S[-1]
        else:
            self.current_damping = base_damping

        # Reset L for inverse computations
        self.L = None

    def _matmul_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Nyström preconditioner to the input tensor x."""
        S_safe = self.S if x.ndim == 1 else self.S.unsqueeze(-1)
        return self.U @ (S_safe * (self.U.T @ x)) + self.current_damping * x

    def _inverse_matmul_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse of the Nyström preconditioner to the input tensor x."""
        x_in = x.unsqueeze(-1) if x.ndim == 1 else x
        damping = self.current_damping

        UTx = self.U.T @ x_in

        # If we are not in double precision, we try to take a more numerically
        # stable approach that requires an additional Cholesky factorization.
        if self.using_low_precision:
            if self.L is None:
                self.L = torch.linalg.cholesky(
                    damping * torch.diag(self.S**-1) + self.U.T @ self.U,
                )
            L_inv_UTx = torch.linalg.solve_triangular(self.L, UTx, upper=False)
            LT_inv_L_inv_UTx = torch.linalg.solve_triangular(
                self.L.T, L_inv_UTx, upper=True
            )
            x_in = 1 / damping * (x_in - self.U @ LT_inv_L_inv_UTx)
        else:
            x_in = 1 / damping * (x_in - self.U @ UTx) + self.U @ torch.divide(
                UTx, (self.S + damping).unsqueeze(-1)
            )

        return x_in.squeeze(-1) if x.ndim == 1 else x_in


def _generate_ortho_embedding(
    dimension: int, sketch_size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Generate an orthogonal random embedding matrix."""
    # Generate a random Gaussian matrix
    Omega = torch.linalg.qr(
        torch.randn(dimension, sketch_size, dtype=dtype, device=device),
        mode="reduced",
    )[0]

    return Omega


def _randomized_power_err_est(
    A: torch.Tensor | LinearOperator,
    U: torch.Tensor,
    S: torch.Tensor,
    num_iters: int,
    dtype: torch.dtype,
) -> float:
    """Estimate approximation error of the Nyström method."""
    v_prev = torch.randn(A.shape[0], dtype=dtype, device=A.device)
    v_prev /= torch.linalg.norm(v_prev)
    err_est = torch.inf

    for _ in range(num_iters):
        v_next = A @ v_prev - U @ (S * (U.T @ v_prev))
        err_est = torch.inner(v_prev, v_next).item()
        v_next /= torch.linalg.norm(v_next)
        v_prev = v_next

    return err_est
