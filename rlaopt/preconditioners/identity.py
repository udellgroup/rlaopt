from rlaopt.preconditioners.preconditioner import Preconditioner
from rlaopt.preconditioners.configs import IdentityConfig


class Identity(Preconditioner):
    def __init__(self, config: IdentityConfig):
        super().__init__(config)

    def _update(self, A):
        pass

    def __matmul__(self, x):
        return x

    def _inverse_matmul(self, x):
        return x
