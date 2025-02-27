import torch

from rlaopt.preconditioners.preconditioner import Preconditioner
from rlaopt.preconditioners.configs import NystromConfig


class Nystrom(Preconditioner):
    def __init__(self, config: NystromConfig):
        super().__init__(config)
        self.U = None
        self.S = None

    def _update(self, A, device):
        # Compute sketching matrix
        Omega = torch.randn(A.shape[1], self.config.rank, device=device)
        Omega = torch.linalg.qr(Omega, mode="reduced")[0]

        # Compute core matrix
        Y = A @ Omega
        Core = Omega.T @ Y
        shift = torch.finfo(Omega.dtype).eps * torch.trace(Core)
        Core.diagonal().add_(shift)

        # Compute Nyström approximation
        L = torch.linalg.cholesky(
            Core,
            upper=False,
        )
        B = torch.linalg.solve_triangular(L, Y.T, upper=False)
        self.U, self.S, _ = torch.linalg.svd(B.T, full_matrices=False)
        self.S = torch.maximum(self.S**2 - shift, torch.tensor(0.0))

    def __matmul__(self, x):
        return self.U @ (self.S * (self.U.T @ x)) + self.config.rho * x

    def _inverse_matmul(self, x):
        UTx = self.U.T @ x
        x = 1 / self.config.rho * (x - self.U @ UTx) + self.U @ torch.divide(
            UTx, self.S + self.config.rho
        )
        return x
