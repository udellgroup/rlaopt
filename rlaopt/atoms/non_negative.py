import torch

from rlaopt.atoms.box import Box
from rlaopt.expression.expression import Variable

class NonNegative(Box):
    def __init__(self, x: Variable):
        l = torch.zeros_like(x.value.data, device=x.value.device)
        super().__init__(x, l, u=None)