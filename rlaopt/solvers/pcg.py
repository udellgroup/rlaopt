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
        # Get current mask from the system
        mask = self.system.mask

        # If all components have converged, nothing to do
        if not mask.any():
            return

        # Apply mask to relevant matrices to only work with non-converged components
        P_masked = self.P_[:, mask]
        RZ_masked = self.RZ[mask][:, mask]  # This is a smaller matrix now

        # Compute A*P only for non-converged directions
        AP_masked = self.system.A @ P_masked + self.system.reg * P_masked

        # Compute alpha for active components
        alpha_masked = torch.linalg.solve(P_masked.T @ AP_masked, RZ_masked)

        # Only update the active parts
        self._W[:, mask] += P_masked @ alpha_masked

        # Update residual for all components
        self.R[:, mask] -= AP_masked @ alpha_masked

        # Compute new preconditioned residuals for active components
        Z_new_masked = self.P._inv @ self.R[:, mask]

        # Update Z with the new values for active components
        self.Z[:, mask] = Z_new_masked

        # Compute new RZ for active components
        RZ_new_masked = self.R[:, mask].T @ Z_new_masked

        # Compute beta for active components
        beta_masked = torch.linalg.solve(RZ_masked, RZ_new_masked)

        # Update P for active components
        self.P_[:, mask] = Z_new_masked + P_masked @ beta_masked

        # Update RZ for the next iteration
        if mask.any():  # Only if there are still active components
            # Create a new RZ tensor with zeros
            new_RZ = torch.zeros_like(self.RZ)
            # Fill in the active part
            new_RZ[torch.outer(mask, mask)] = RZ_new_masked.flatten()
            self.RZ = new_RZ
        else:
            # All components have converged, set RZ to zeros
            self.RZ = torch.zeros_like(self.RZ)
