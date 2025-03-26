from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from rlaopt.solvers.solver import Solver
from rlaopt.solvers.configs import SAPAccelConfig
from rlaopt.preconditioners import Preconditioner, PreconditionerConfig
from rlaopt.preconditioners import _get_precond as _pf_get_precond

if TYPE_CHECKING:
    from rlaopt.models import LinSys  # Import only for type hints


class SAP(Solver):
    def __init__(
        self,
        system: "LinSys",
        w_init: torch.Tensor,
        precond_config: PreconditionerConfig,
        device: torch.device,
        blk_sz: int,
        accel: bool,
        accel_config: Optional[SAPAccelConfig],
        power_iters: int,
    ):
        self.system = system
        self.precond_config = precond_config
        self._w = w_init.clone()
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

            self.v = self._w.clone()
            self.y = self._w.clone()

    @property
    def w(self):
        return self._w

    def _get_precond(self, blk: torch.Tensor):
        P = _pf_get_precond(self.precond_config)
        P._update(self.system.A_blk_oracle(blk), self.device)
        return P

    def _get_blk(self):
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

    def _get_stepsize(self, blk: torch.Tensor, blk_precond: Preconditioner):
        v = torch.randn(blk.shape[0], device=self.device)
        v /= torch.linalg.norm(v)

        max_eig = None

        # Randomized power iteration to estimate the maximum eigenvalue
        for _ in range(self.power_iters):
            v_old = v.clone()
            v = blk_precond._inv @ v
            v = self.system.A_blk_oracle(blk) @ v + self.system.reg * v
            max_eig = torch.dot(v, v_old)
            v /= torch.linalg.norm(v)
        return max_eig ** (-1.0)

    def _get_damping(self, blk: torch.Tensor, blk_precond: Preconditioner):
        v = torch.randn(blk.shape[0], device=self.device)
        v /= torch.linalg.norm(v)

        max_eig = None

        # Randomized power iteration to estimate the maximum eigenvalue
        for _ in range(self.power_iters):
            v_old = v.clone()
            v = (
                self.system.A_blk_oracle(blk) @ v
                + self.precond_config.rho * v
                - blk_precond @ v
            )
            max_eig = torch.dot(v, v_old)
            v /= torch.linalg.norm(v)

        self.precond_config.rho = max_eig

    def _get_block_update(
        self, w: torch.Tensor, blk: torch.Tensor, blk_precond: Preconditioner
    ):
        # Compute the block gradient
        blk_grad = (
            self.system.A_row_oracle(blk) @ w
            + self.system.reg * w[blk]
            - self.system.b[blk]
        )

        # Apply the preconditioner
        dir = blk_precond._inv @ blk_grad
        return dir

    def _step(self):
        # Randomly select a block
        blk = self._get_blk()

        # Compute block preconditioner and learning rate
        blk_precond = self._get_precond(blk)
        if self.precond_config.damping_strategy == "adaptive":
            self._get_damping(blk, blk_precond)
            blk_stepsize = 0.5

        else:
            blk_stepsize = self._get_stepsize(blk, blk_precond)

        # Get the update direction
        # Update direction is computed at self.y if accelerated, else at self._w
        eval_loc = self.y if self.accel else self._w
        dir = self._get_block_update(eval_loc, blk, blk_precond)

        # Update parameters
        if self.accel:
            self._w = self.y.clone()
            self._w[blk] -= blk_stepsize * dir
            self.v = self.beta * self.v + (1 - self.beta) * self.y
            self.v[blk] -= blk_stepsize * self.gamma * dir
            self.y = self.alpha * self.v + (1 - self.alpha) * self._w
        else:
            self._w[blk] -= blk_stepsize * dir
