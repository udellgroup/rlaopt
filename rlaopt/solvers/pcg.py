import torch

from rlaopt.models import LinSys
from rlaopt.preconditioners import Nystrom


class PCG:
    def __init__(self, system: LinSys, w_init: torch.Tensor, precond_params: dict):
        self.system = system
        self.precond_params = precond_params
        self.w = w_init.clone()
        self.P = self._get_precond()
        self.r, self.z, self.p, self.rz = self._init_pcg()

    def _init_pcg(self):
        r = self.system.b - (self.system.A @ self.w + self.system.reg * self.w)
        z = self.P._inv @ r
        p = z.clone()
        rz = torch.dot(r, z)
        return r, z, p, rz

    def _get_precond(self):
        if self.precond_params["type"] == "nystrom":
            P = Nystrom(self.precond_params["params"])
        else:
            raise ValueError("Invalid preconditioner type")
        P._update(self.system.A)
        return P

    def _step(self):
        Ap = self.system.A @ self.p + self.system.reg * self.p
        alpha = self.rz / torch.dot(Ap, self.p)
        self.w += alpha * self.p
        self.r -= alpha * Ap
        self.z = self.P._inv @ self.r
        rz_new = torch.dot(self.r, self.z)
        self.p = self.z + (rz_new / self.rz) * self.p
        self.rz = rz_new
