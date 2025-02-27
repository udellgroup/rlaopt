from rlaopt.preconditioners.sketches.sketch import Sketch

import torch


class Gauss(Sketch):
    def __init__(self, mode, sketch_size, matrix_dim, device):
        super().__init__(mode, sketch_size, matrix_dim, device)

    def _generate_embedding(self):
        # TODO: Add comment discussing our normalization convention
        Smat = torch.randn(self.s, self.d, device=self.device) / (self.s) ** (0.5)
        if self.mode == "right":
            Smat = Smat.T
        return Smat
