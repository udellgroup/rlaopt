from typing import TYPE_CHECKING

import torch

from rlaopt.solvers.solver import Solver
from rlaopt.preconditioners import Nystrom, Identity
from rlaopt.preconditioners import (
    PreconditionerConfig,
    NystromConfig,
    IdentityConfig,
)

if TYPE_CHECKING:
    from rlaopt.models.linsys import LinSys  # Import only for type hints


class PCG(Solver):
    def __init__(
        self,
        system: "LinSys",
        w_init: torch.Tensor,
        precond_config: PreconditionerConfig,
        device: torch.device,
    ):
        self.system = system
        self.precond_config = precond_config
        self._w = w_init.clone()
        self.device = device
        self.P = self._get_precond()
        self.r, self.z, self.p, self.rz = self._init_pcg()

    @property
    def w(self):
        return self._w

    def _init_pcg(self):
        r = self.system.b - (self.system.A @ self._w + self.system.reg * self._w)
        z = self.P._inv @ r
        p = z.clone()
        rz = torch.dot(r, z)
        return r, z, p, rz

    def _get_precond(self):
        if isinstance(self.precond_config, NystromConfig):
            P = Nystrom(self.precond_config)
        elif isinstance(self.precond_config, IdentityConfig):
            P = Identity(self.precond_config)
        else:
            raise ValueError("Invalid preconditioner type")
        print(self.system.A.shape)
        P._update(self.system.A, self.device)
        return P

    def _step(self):
        Ap = self.system.A @ self.p + self.system.reg * self.p
        alpha = self.rz / torch.dot(Ap, self.p)
        self._w += alpha * self.p
        self.r -= alpha * Ap
        self.z = self.P._inv @ self.r
        rz_new = torch.dot(self.r, self.z)
        self.p = self.z + (rz_new / self.rz) * self.p
        self.rz = rz_new
