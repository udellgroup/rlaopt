import rlaopt.atoms
import torch

class Leaf(rlaopt.atoms.Atom):
    def __init__(self, ...):
        ...

class SumSquared(rlaopt.atoms.Atom):
    def __init__(self, arg | torch.Tensor | atom):
        super().__init__()
        self.arg = rlaopt.utils.to_atom(arg)

    def _forward_impl(self) -> torch.Tensor:
        val = self.arg()
        return (val**2).sum()

    def is_smooth(self):
        return self.arg.is_smooth()

    def is_proxable(self):
        return self.arg.is_affine()

    def prox(self, scale):
        if self.arg.is_leaf():
            return 1 / (1 + scale * self.scaling) * self.arg
        else:
            raise NotImplementedError # Do something with NysCG


"""
x = SumSquared(X @ variable - y)
y = x.copy()

# Inputs
# Parameters

||Xb - y||_2^2
 - b is a torch.nn.parameters.Parameter
 - X, y are inputs


"Linear fit"
class LinearFit(Atom):
    def __init__(self, dataloader, param):
        self.... = ...

    def _forward_impl(self):
        return torch.vstack(
            [X @ self.param - y for X, y in self.dataloader])

class quad_form(Atom):
    def __init__(self, param, tensor):
        self.... = ...


Parameter
Variable
||X b - y||_2^2
 - X, y are parameters are effectively the same as a "Constant"
 - b is a variable

x^T P x = cp.quad_form(x, P)
 - P is a parameter or a constant
 - x is a variable

"""
