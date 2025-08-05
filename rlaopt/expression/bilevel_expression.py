from typing import Callable

import torch

from .expression import Expression
from ..solvers.configs import ProxGradConfig
from ..solvers.proximal_gradient.prox_grad import ProximalGradient

SolverConfig = ProxGradConfig
Solver = ProximalGradient

class BilevelExpression(Expression):
    """
    An Expression class that represents a bilevel optimization problem.

    This class encapsulates a bilevel optimization problem where the inner problem is solved
    using a solver. The outer problem is defined by the expression `F_out`, and the inner problem
    is defined by the expression `F_in`. The inner problem is parameterized by `w`, which is a learnable parameter.
    """
    def __init__(
            self, 
            w0: torch.Tensor, 
            F_in: Callable[[torch.Tensor|torch.nn.Parameter], Expression], 
            F_out: Expression,
            solver_config: SolverConfig,
            solver: Solver):
        """Initializes the bilevel expression.
        
        Args:
            w0: Initial value for the parameter `w`.
            F_in: Function that takes `w` and returns an expression for the inner problem.
            F_out: Expression representing the outer problem.
            solver_config: Configuration for the solver used to solve the inner problem.
            solver: Solver instance used to solve the inner problem.
        """
        super().__init__()
        object.__setattr__(self, 'F_in', F_in)
        object.__setattr__(self, 'F_out', F_out)
        self.solver =  lambda x:  solver(x, solver_config)
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
    
    def prox(self):
        raise NotImplementedError
    
    def to_cvxpy(self):
        raise NotImplementedError