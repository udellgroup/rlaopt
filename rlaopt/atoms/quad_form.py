import torch
from linops import LinearOperator, MatrixOperator

from rlaopt.atoms.atom import Atom
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


class QuadForm(Atom):
    def __init__(
        self,
        x: Variable,
        A: torch.Tensor | LinearOperator,
        b: TensorDict,
        x_c: torch.Tensor | None = None,
    ):
        if isinstance(A, torch.Tensor):
            A = MatrixOperator(A, A)

        b = b.to_flat_tensor()

        exprs = {"x": x}
        if x_c is None:
            x_c = torch.zeros(1, device=b.device, dtype=b.dtype)
        buffers = {"b": b, "x_c": x_c}
        variable_names = ["x"]
        super().__init__(exprs, buffers, variable_names)
        self.A = A
        self.A.requires_grad_(False)

    def forward(self):
        input_tensor = self.get_input("x").forward()
        # x - x_c
        d = input_tensor - self.get_buffer("x_c")
        # <b, x - x_c> + 0.5 * <x-x_c, A(x-x_c)>
        value = torch.dot(self.get_buffer("b"), d) + 0.5 * torch.dot(d, self.A @ d)
        return value
