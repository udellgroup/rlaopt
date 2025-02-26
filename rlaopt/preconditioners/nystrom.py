import torch

from rlaopt.preconditioners.preconditioner import Preconditioner


class Nystrom(Preconditioner):
    def __init__(self, params):
        super().__init__(params)
        self.U = None
        self.S = None

    def _update(self, A):
        # Compute sketching matrix
        Omega = torch.randn(
            A.shape[1], self.params["rank"], device=self.params["device"]
        )
        Omega = torch.linalg.qr(Omega, mode="reduced")[0]

        # Compute core matrix
        Y = A @ Omega
        Core = Omega.T @ Y
        shift = torch.finfo(Omega.dtype).eps * torch.trace(Core)

        # Compute Nyström approximation
        L = torch.linalg.cholesky(
            Core.diagonal().add_(shift),
            upper=False,
        )
        B = torch.linalg.solve_triangular(L, Y.T, upper=False)
        self.U, self.S, _ = torch.linalg.svd(B.T, full_matrices=False)
        self.S = torch.maximum(self.S**2 - shift, torch.tensor(0.0))

    def __matmul__(self, x):
        return self.U @ (self.S * (self.U.T @ x)) + self.params["rho"] * x

    def _inverse_matmul(self, x):
        UTx = self.U.T @ x
        x = 1 / self.params["rho"] * (x - self.U @ UTx) + self.U @ torch.divide(
            UTx, self.S + self.params["rho"]
        )
        return x
