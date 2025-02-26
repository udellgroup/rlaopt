from rlaopt.preconditioners.preconditioner import Preconditioner


class Identity(Preconditioner):
    def __init__(self, params):
        super().__init__(params)

    def _update(self, A):
        pass

    def __matmul__(self, x):
        return x

    def _inverse_matmul(self, x):
        return x
