import torch

from rlaopt.preconditioners.preconditioner import Preconditioner


class Nystrom(Preconditioner):
    def __init__(self, params):
        super().__init__(params)

        self.U = None
        self.S = None

    def _check_inputs(self, params):
        # Check that "rank" and "rho" are in the params dictionary
        if "rank" not in params:
            raise ValueError("params dictionary must contain 'rank'")
        if "rho" not in params:
            raise ValueError("params dictionary must contain 'rho'")
        if "device" not in params:
            raise ValueError("params dictionary must contain 'device'")
        # Check that "rank" and "rho" are valid
        if not isinstance(params["rank"], int) or params["rank"] <= 0:
            raise ValueError("rank must be a positive integer")
        if not isinstance(params["rho"], float) or params["rho"] < 0:
            raise ValueError("rho must be a non-negative float")
        if not isinstance(params["device"], torch.device):
            raise ValueError("device must be a torch.device object")

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
            Core + shift * torch.eye(self.params["rank"], device=self.params["device"]),
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
