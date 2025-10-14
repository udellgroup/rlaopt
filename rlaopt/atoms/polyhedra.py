from functools import partial

from typing import Callable

import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable


class Polyhedra(AtomExpression):
    def __init__(
        self,
        x: Variable,
        A: torch.Tensor = None,
        b: torch.Tensor = None,
        C: torch.Tensor = None,
        l: torch.Tensor = None,
        u: torch.Tensor = None,
    ):
        super().__init__()
        if (A is not None) and (b is None):
            raise ValueError("b cannot be None when A is not None")

        # Validate input dimensional consistency
        _validate(A, C, b, l, u)

        # if u is provided but not l, set l to -infinity
        if (u is not None) and (l is None):
            l = torch.tensor(-torch.inf, device=u.device, dtype=u.dtype)
        # if l is provided but not u, set u to infinity
        elif (l is not None) and (u is None):
            u = torch.tensor(torch.inf, device=l.device, dtype=l.dtype)

        # Register the variable as a parameter
        self.register_variable(x)

        # Register constraint data as buffers
        if (A is not None) and (b is not None):
            self.register_atom_buffer("A", A)
            self.register_atom_buffer("b", b)
        else:
            self.A = None
            self.b = None

        if C is not None:
            self.register_atom_buffer("C", C)
        else:
            self.C = None

        if l is not None:
            self.register_atom_buffer("l", l)
            self.register_atom_buffer("u", u)
        else:
            self.u = None
            self.l = None

        # build evaluation function
        self._eval = _build_eval(self.A, self.C, self.b, self.l, self.u)

    def forward(self) -> torch.Tensor:
        value = self.get_variable(self.var_name)
        return self._eval(value)

    def is_smooth(self):
        return False

    def is_proxable(self):
        return False

    def is_subsamplable(self):
        return False

    def subsample(self, indices):
        raise NotImplementedError("Polyhedra is not subsamplable")

    def to_cvxpy(self):
        return super().to_cvxpy()


def _validate(A, C, b, l, u):
    if A is not None and b is not None:
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b must have matching row counts")
    if C is not None:
        if (l is not None and C.shape[0] != l.shape[0]) or (
            u is not None and C.shape[0] != u.shape[0]
        ):
            raise ValueError("C, l, and u must have matching row counts")


def _build_eval(A, C, b, l, u) -> Callable[[torch.Tensor], torch.Tensor]:
    eq_exists = A is not None and b is not None
    ineq_exists = l is not None  # implies u is not None

    eval_fns = []

    if eq_exists:
        if A.dim() > 1:
            eval_fns.append(partial(_eval_eq, A=A, b=b))
        else:
            eval_fns.append(partial(_eval_hyperplane, a=A, b=b))

    if ineq_exists:
        if C is not None:
            if C.dim() > 1:
                eval_fns.append(partial(_eval_ineq, C=C, l=l, u=u))
            else:
                eval_fns.append(partial(_eval_halfspace, c=C, l=l, u=u))
        else:
            eval_fns.append(partial(_eval_id_ineq, l=l, u=u))

    if not eval_fns:
        raise ValueError(
            "Provided constraints define a trivial polyhedron (no constraints)."
        )

    def _eval(x: torch.Tensor):
        return sum(fn(x) for fn in eval_fns)

    return _eval


def _eval_id_ineq(x: torch.Tensor, l: torch.Tensor, u: torch.Tensor):
    statement = (l <= x) & (x <= u)
    return _indicator(statement)


def _eval_ineq(x: torch.Tensor, C: torch.Tensor, l: torch.Tensor, u: torch.Tensor):
    statement = (l <= C @ x) & (C @ x <= u)
    return _indicator(statement)


def _eval_halfspace(x: torch.Tensor, c: torch.Tensor, l: torch.Tensor, u: torch.Tensor):
    statement = (l <= torch.dot(c, x)) & (torch.dot(c, x) <= u)
    return _indicator(statement)


def _eval_eq(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor):
    statement = A @ x == b
    return _indicator(statement)


def _eval_hyperplane(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    statement = torch.dot(a, x) == b
    return _indicator(statement)


def _indicator(statement: torch.Tensor):
    if statement.all():
        return torch.tensor(0.0, device=statement.device)
    else:
        return torch.tensor(torch.inf, device=statement.device)
