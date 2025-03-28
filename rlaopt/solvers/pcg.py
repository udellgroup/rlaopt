from typing import TYPE_CHECKING

import torch

from .solver import Solver
from rlaopt.preconditioners import PreconditionerConfig
from rlaopt.preconditioners import _get_precond as _pf_get_precond

if TYPE_CHECKING:
    from rlaopt.models import LinSys  # Import only for type hints


class PCG(Solver):
    def __init__(
        self,
        system: "LinSys",
        W_init: torch.Tensor,
        precond_config: PreconditionerConfig,
        device: torch.device,
    ):
        self.system = system
        self.precond_config = precond_config
        self._W = W_init.clone()
        self.device = device
        self.P = self._get_precond()
        self.R, self.Z, self.P_, self.RZ = self._init_pcg()

    @property
    def W(self):
        return self._W

    def _init_pcg(self):
        R = self.system.B - (self.system.A @ self._W + self.system.reg * self._W)
        Z = self.P._inv @ R
        P_ = Z.clone()
        RZ = R.T @ Z
        return R, Z, P_, RZ

    def _get_precond(self):
        P = _pf_get_precond(self.precond_config)
        P._update(self.system.A, self.device)
        P._update_damping(baseline_rho=self.system.reg)
        return P

    def _step(self):
        AP_ = self.system.A @ self.P_ + self.system.reg * self.P_
        L = torch.linalg.cholesky(self.P_.T @ AP_, upper=False)
        alpha = torch.linalg.solve_triangular(L, self.RZ, upper=False)
        alpha = torch.linalg.solve_triangular(L.T, alpha, upper=True)
        self._W += self.P_ @ alpha
        self.R -= AP_ @ alpha

        self.Z = self.P._inv @ self.R
        L = torch.linalg.cholesky(self.RZ, upper=False)
        RZ_new = self.R.T @ self.Z
        beta = torch.linalg.solve_triangular(L, RZ_new, upper=False)
        beta = torch.linalg.solve_triangular(L.T, beta, upper=True)
        self.P_ = self.Z + self.P_ @ beta
        self.RZ = RZ_new
