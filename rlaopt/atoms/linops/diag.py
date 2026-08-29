from .base_linop import Linop

class DaigonalLinop(Linop):
    def __init__(self, diag, input_):
        super().__init__((diag.shape[0], diag.shape[0]))
        self._diag = diag
        self.register_input(input_)

    def forward(self):
        return self.get_input().forward() * self._diag

