from __future__ import annotations

import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable


class NucNorm(AtomExpression):
    def __init__(
        self, x: Variable, scaling: float | torch.Tensor | torch.nn.Parameter = 1.0
    ):
        super().__init__()

        if not isinstance(x, Variable):
            raise TypeError(f"Expected Variable, got {type(x)}")

        if x.value.data.dim() != 2:
            raise ValueError(
                f"Variable value must be 2D Tensor, but got {x.value.data.dim()}D Tensor."
            )

        # Register the variable as a parameter
        self.register_variable(x)

        # Register the scaling factor
        self.register_atom_buffer("scaling", scaling)

    def is_smooth(self):
        return False

    def forward(self):
        """Evaluates the nuclear norm at the registered variable value."""
        value = self.get_variable(self.var_name)
        S = torch.linalg.svdvals(value)
        return self.scaling * torch.sum(S)

    def is_proxable(self):
        return True

    def prox(self, location, prox_scaling):
        U, S, Vt = torch.linalg.svd(location, full_matrices=False)
        S = torch.nn.functional.relu(S - prox_scaling * self.scaling)
        return (U * S) @ Vt

    def is_subsamplable(self):
        return False

    def subsample(self, indices):
        raise NotImplementedError("Nuclear norm cannot be subsampled")

    def to_cvxpy(self):
        raise NotImplementedError("NucNorm does not support conversion to cvxpy")
