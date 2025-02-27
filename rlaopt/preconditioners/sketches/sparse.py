from rlaopt.preconditioners.sketches.sketch import Sketch

import torch


class Sparse(Sketch):
    def __init__(self, mode, sketch_size, matrix_dim, device):
        super().__init__(mode, sketch_size, matrix_dim, device)

    def _generate_embedding(self):
        zeta = 8 if self.s >= 8 else self.s

        # Initialize S as a zero matrix
        Smat = torch.zeros((self.s, self.d), device=self.device)

        # Generate random +1/-1 values for zeta entries in each column
        b = torch.bernoulli(
            0.5 * torch.ones((zeta, self.n), device=self.device)
        )  # Bernoulli(0.5)
        z = 2 * b - 1  # Convert to +1/-1

        # Generate random row indices for each non-zero entry in each column
        row_indices = torch.randint(
            self.s, (zeta, self.n), device=self.device
        )  # Random row indices

        # Scatter update: place z values in S at the appropriate row indices
        Smat.scatter_(0, row_indices, z)  # In-place scatter

        # Scale S
        Smat = Smat * torch.sqrt(
            1 / torch.tensor(zeta, dtype=torch.float, device=self.device)
        )
        if self.mode == "right":
            Smat = Smat.T

        return Smat
