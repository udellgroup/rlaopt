from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from .solver import Solver
from .configs import SAPAccelConfig
from rlaopt.linops import LinOp
from rlaopt.preconditioners import (
    Preconditioner,
    PreconditionerConfig,
    IdentityConfig,
    NewtonConfig,
    NystromConfig,
)
from rlaopt.preconditioners import _get_precond as _pf_get_precond
from rlaopt.spectral_estimators import randomized_powering

if TYPE_CHECKING:
    from rlaopt.models import LinSys  # Import only for type hints

VALID_PRECONDS = [IdentityConfig, NewtonConfig, NystromConfig]


class SAP(Solver):
    def __init__(
        self,
        system: "LinSys",
        W_init: torch.Tensor,
        precond_config: PreconditionerConfig,
        device: torch.device,
        blk_sz: int,
        accel: bool,
        accel_config: Optional[SAPAccelConfig],
        power_iters: int,
    ):
        self.system = system

        # Check if preconditioner is valid
        if type(precond_config) not in VALID_PRECONDS:
            raise TypeError(
                f"Valid preconditioner configs for SAP are {VALID_PRECONDS}, "
                f"but received {type(precond_config)}"
            )
        self.precond_config = precond_config

        self._W = W_init.clone()
        self.device = device
        self.blk_sz = blk_sz
        self.accel = accel
        self.accel_config = accel_config
        self.power_iters = power_iters

        self.probs = torch.ones(self.system.A.shape[0]) / self.system.A.shape[0]
        self.probs_cpu = self.probs.cpu().numpy()

        # Setup acceleration parameters
        if self.accel:
            self.beta = 1 - (self.accel_config.mu / self.accel_config.nu) ** 0.5
            self.gamma = 1 / (self.accel_config.mu * self.accel_config.nu) ** 0.5
            self.alpha = 1 / (1 + self.gamma * self.accel_config.nu)

            self.V = self._W.clone()
            self.Y = self._W.clone()

    @property
    def W(self):
        return self._W

    def _get_precond(self, blk: torch.Tensor) -> Preconditioner:
        P = _pf_get_precond(self.precond_config)
        P._update(self.system.A_blk_oracle(blk), self.device)
        P._update_damping(baseline_rho=self.system.reg)
        return P

    def _get_blk(self) -> torch.Tensor:
        try:
            blk = torch.multinomial(self.probs, self.blk_sz, replacement=False)
        except RuntimeError as e:
            if "number of categories cannot exceed" not in str(e):
                raise e
            blk = np.random.choice(
                self.probs.shape[0], size=self.blk_sz, replace=False, p=self.probs_cpu
            )
            blk = torch.from_numpy(blk)
        return blk

    def _get_stepsize(self, blk: torch.Tensor, blk_precond: Preconditioner) -> float:
        if isinstance(self.precond_config, NewtonConfig):
            # If the damping is the regularization, then the preconditioner is exact
            # and the stepsize is 1.0
            if self.precond_config.rho == self.system.reg:
                return 1.0
        else:
            # If the preconditioner is not exact, we need to compute the stepsize
            def blk_matvec(v):
                return self.system.A_blk_oracle(blk) @ v + self.system.reg * v

            # Note that S does not correspond to a symmetric matrix.
            # This is ok because we assume that A is symmetric,
            # which also implies the preconditioner P is symmetric.
            # It follows that P^(-1) A can be plugged into randomized_powering
            # and the result will be the same as if we had used P^(-1/2) A P^(-1/2).
            S = LinOp(
                device=self.device,
                shape=torch.Size((self.blk_sz, self.blk_sz)),
                matvec=blk_precond._inverse_matmul_compose(blk_matvec),
            )

        max_eig, _ = randomized_powering(S, max_iters=self.power_iters)
        return max_eig ** (-1.0)

    def _get_block_update(
        self,
        W: torch.Tensor,
        B: torch.Tensor,
        blk: torch.Tensor,
        blk_precond: Preconditioner,
    ):
        # Compute the block gradient
        blk_grad = (
            self.system.A_row_oracle(blk) @ W + self.system.reg * W[blk, :] - B[blk, :]
        )

        # Apply the preconditioner
        dir = blk_precond._inv @ blk_grad
        return dir

    def _step(self):
        # Get mask
        mask = self.system.mask

        # If all components have converged, nothing to do
        if not mask.any():
            return

        # Randomly select a block
        blk = self._get_blk()

        # Compute block preconditioner and learning rate
        blk_precond = self._get_precond(blk)
        blk_stepsize = self._get_stepsize(blk, blk_precond)

        # Get the update direction
        # Update direction is computed at self.Y if accelerated, else at self._W
        eval_loc = self.Y[:, mask] if self.accel else self._W[:, mask]
        dir = self._get_block_update(eval_loc, self.system.B[:, mask], blk, blk_precond)

        # Update
        if self.accel:
            # Copy accelerated point to solution for masked columns
            self._W[:, mask] = self.Y[:, mask].clone()

            # Create update and apply it
            update = torch.zeros_like(self._W[:, mask])
            update[blk] = blk_stepsize * dir
            self._W[:, mask] -= update

            # Update momentum terms similarly
            self.V[:, mask] = (
                self.beta * self.V[:, mask] + (1 - self.beta) * self.Y[:, mask]
            )

            v_update = torch.zeros_like(self.V[:, mask])
            v_update[blk] = blk_stepsize * self.gamma * dir
            self.V[:, mask] -= v_update

            # Update acceleration point
            self.Y[:, mask] = (
                self.alpha * self.V[:, mask] + (1 - self.alpha) * self._W[:, mask]
            )
        else:
            update = torch.zeros_like(self._W[:, mask])
            update[blk] = blk_stepsize * dir
            self._W[:, mask] -= update
