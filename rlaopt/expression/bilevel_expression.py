from typing import Callable

import cvxpy as cp
import torch

from rlaopt.expression.expression import Expression
from rlaopt.solvers.configs_base import SolverConfig
from rlaopt.solvers.solver_base import OptimSolver


class BilevelExpression(Expression):
    """An Expression class that represents a bilevel optimization problem.

    This class encapsulates a bilevel optimization problem where
    the inner problem is solved using a solver.
    The outer problem is defined by the expression `F_out`,
    and the inner problem is defined by the expression `F_in`.
    The inner problem is parameterized by `w`, which is a learnable parameter.
    """

    def __init__(
        self,
        w0: torch.Tensor,
        F_in: Callable[[torch.Tensor], Expression],
        F_out: Expression,
        solver_config: SolverConfig,
        solver: OptimSolver,
    ):
        """Initializes the bilevel expression.

        Args:
            w0: Initial value for the parameter `w`.
            F_in: Function that takes `w` and returns an expression
              for the inner problem.
            F_out: Expression representing the outer problem.
            solver_config: Configuration for the solver used to solve the inner problem.
            solver: Solver instance used to solve the inner problem.
        """
        super().__init__()
        # TODO(pratik): we use __setattr__ to avoid the parameters in F_in and F_out
        # being automatically registered as parameters of this class.
        # This is a workaround, and we should ideally refactor the code to avoid this,
        # since this could make it difficult to move instances of this class
        # between devices (e.g., CPU to GPU).
        object.__setattr__(self, "F_in", F_in)
        object.__setattr__(self, "F_out", F_out)
        self.solver = lambda x: solver(solver_config, x)
        self.w = torch.nn.Parameter(w0, requires_grad=True)

    def evaluate_at(self):
        obj_in = self.F_in(self.w)
        params, _ = self.solver(obj_in).solve(obj_in)
        params = self.F_out.expr_convert_params(params)
        return self.F_out.evaluate(params)

    def is_smooth(self):
        return True

    def is_proxable(self):
        raise NotImplementedError

    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        raise NotImplementedError("BilevelExpression is not proxable")

    def to_cvxpy(self) -> cp.Expression:
        raise NotImplementedError("BilevelExpression cannot be converted to CVXPY")
