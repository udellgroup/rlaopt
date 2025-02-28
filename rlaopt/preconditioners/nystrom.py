import torch

from rlaopt.preconditioners.preconditioner import Preconditioner
from rlaopt.preconditioners.configs import NystromConfig
from rlaopt.preconditioners.sketches.gauss import Gauss
from rlaopt.preconditioners.sketches.sparse import Sparse
from rlaopt.preconditioners.sketches.ortho import Ortho


class Nystrom(Preconditioner):
    def __init__(self, config: NystromConfig):
        super().__init__(config)
        self.U = None
        self.S = None

    def _update(self, A, device: torch.device) -> None:
        # TODO think about making a Sketches factory to make this more modular
        # Compute sketching matrix
        if self.config.sketch == "gauss":
            Omega = Gauss("right", self.config.rank, A.shape[1], device=device)
            # Omega = torch.randn(A.shape[1], self.config.rank, device=device)
            # Omega = torch.linalg.qr(Omega, mode="reduced")[0]
        elif self.config.sketch == "ortho":
            Omega = Ortho("right", self.config.rank, A.shape[1], device=device)
        else:
            Omega = Sparse("right", self.config.rank, A.shape[1], device=device)

        # Compute core matrix
        # Y = A @ Omega
        Y = Omega._apply_right(A)
        Core = Omega._apply_left_trans(Y)
        shift = torch.finfo(Y.dtype).eps * torch.trace(Core)
        Core.diagonal().add_(shift)

        # Compute Nyström approximation
        L = torch.linalg.cholesky(
            Core,
            upper=False,
        )
        B = torch.linalg.solve_triangular(L, Y.T, upper=False)
        self.U, self.S, _ = torch.linalg.svd(B.T, full_matrices=False)
        self.S = torch.maximum(self.S**2 - shift, torch.tensor(0.0))

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.U @ (self.S * (self.U.T @ x)) + self.config.rho * x

    def _inverse_matmul(self, x: torch.Tensor) -> torch.Tensor:
        UTx = self.U.T @ x
        x = 1 / self.config.rho * (x - self.U @ UTx) + self.U @ torch.divide(
            UTx, self.S + self.config.rho
        )
        return x
