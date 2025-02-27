import torch

from rlaopt.preconditioners.preconditioner import Preconditioner
from rlaopt.preconditioners.configs import NewtonConfig


class Newton(Preconditioner):
    def __init__(self, config: NewtonConfig):
        super().__init__(config)
        self.L = None

    def _update(self, A, device):
        # Handle the tensor and linear operator cases
        if isinstance(A, torch.Tensor):
            A_true = A
        else:
            A_true = A @ torch.eye(A.shape[1], device=device)
        A_true.diagonal().add_(self.config.rho)  # Add rho to the diagonal in-place
        self.L = torch.linalg.cholesky(A_true, upper=False)

    def __matmul__(self, x):
        return self.L @ (self.L.T @ x)

    def _inverse_matmul(self, x):
        x = torch.linalg.solve_triangular(self.L, x.unsqueeze(-1), upper=False)
        x = torch.linalg.solve_triangular(self.L.T, x, upper=True)
        return x.squeeze()
