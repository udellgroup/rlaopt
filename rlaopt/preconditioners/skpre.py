import torch

from rlaopt.preconditioners.preconditioner import Preconditioner
from rlaopt.preconditioners.configs import SkPreConfig
from rlaopt.preconditioners.sketches.gauss import Gauss
from rlaopt.preconditioners.sketches.sparse import Sparse
from rlaopt.preconditioners.sketches.ortho import Ortho


class SkPre(Preconditioner):
    def __init__(self, config: SkPreConfig):
        super().__init__(config)
        self.Y = None
        self.L = None

    def _update(self, A, device):
        # Get sketching matrix
        if self.config.sketch == "gauss":
            Omega = Gauss("left", self.config.sketch_size, A.shape[1], device=device)
        elif self.config.sketch == "ortho":
            Omega = Ortho("left", self.config.sketch_size, A.shape[1], device=device)
        else:
            Omega = Sparse("left", self.config.sketch_size, A.shape[1], device=device)

        # Compute sketch
        # Y = Omega @ A
        self.Y = Omega._apply_left(A)

        # Construct preconditioner
        if self.config.rho == 0:
            # No damping: get L from Cholesky
            # Delete Y as it won't be needed
            G = self.Y.T @ self.Y
            self.L = torch.linalg.cholesky(G, upper=False)
            self.Y = None
            s_sps = torch.linalg.svdvals(A @ torch.linalg.inv(self.L.T))
            print(
                f"Preconditioned condition number: {s_sps[0].item()/s_sps[-1].item()}"
            )
        else:
            # Damping: Get L via Cholesky
            # Two cases based on sketch size
            # Case 1: sketch_size<= # col(A)
            if self.config.sketch_size >= A.shape[1]:
                G = self.Y.T @ self.Y
                G.diagonal().add_(self.config.rho)
                self.L = torch.linalg.cholesky(G, upper=False)
                self.Y = None
            else:
                # Case 2: sketch_size<# col(A)
                G = self.Y @ self.Y.T
                G.diagonal().add_(self.config.rho)
                self.L = torch.linalg.cholesky(G, upper=False)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        if self.Y is None:
            return self.L @ (self.L.T @ x)
        else:
            return self.Y.T @ (self.Y @ x) + self.config.rho * x

    def _inverse_matmul(self, x: torch.Tensor) -> torch.Tensor:
        if self.Y is None:
            L_inv_x = torch.linalg.solve_triangular(
                self.L.T, x.unsqueeze(-1), upper=True
            )
            P_inv_x = torch.linalg.solve_triangular(self.L, L_inv_x, upper=False)
            return P_inv_x.squeeze()
        else:
            Yx = self.Y @ x
            L_inv_Yx = torch.linalg.solve_triangular(
                self.L, Yx.unsqueeze(-1), upper=False
            )
            LT_inv_L_inv_Yx = torch.linalg.solve_triangular(
                self.L.T, L_inv_Yx, upper=True
            )
            LT_inv_L_inv_Yx = LT_inv_L_inv_Yx.squeeze()
            P_inv_x = 1 / self.config.rho * (x - self.Y.T @ LT_inv_L_inv_Yx)
            return P_inv_x
