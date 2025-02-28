from rlaopt.preconditioners.sketches.sketch import Sketch

import torch


class Ortho(Sketch):
    def __init__(self, mode, sketch_size, matrix_dim, device):
        super().__init__(mode, sketch_size, matrix_dim, device)

    def _generate_embedding(self):
        Smat = torch.linalg.qr(
            torch.randn(self.d, self.s, device=self.device), mode="reduced"
        )[0]
        if self.mode == "left":
            Smat = Smat.T
        return Smat
