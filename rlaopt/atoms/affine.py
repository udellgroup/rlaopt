from __future__ import annotations

import cvxpy as cp
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable


class Affine(AtomExpression):
    def __init__(self, x: Variable, A: torch.Tensor, b: torch.Tensor):
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        self.register_variable(x)
        self.register_atom_buffer("A", A)
        self.register_atom_buffer("b", b)

    def is_smooth(self):
        return True

    def forward(self):
        value = self.get_variable(self.var_name)
        return self.A @ value + self.b

    def is_proxable(self):
        return False

    def prox(self, location, prox_scaling):
        raise NotImplementedError("Affine is not proxable.")

    def is_subsamplable(self):
        return True

    def subsample(self, indices) -> Affine:
        return Affine(getattr(self.var_name), self.A[indices], self.b[indices])

    def to_cvxpy(self) -> cp.Expression:
        raise NotImplementedError("Affine does not yet support CVXPY conversion.")
