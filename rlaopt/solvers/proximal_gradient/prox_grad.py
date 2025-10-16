from typing import Dict, Tuple

import torch
from pydantic import Field

from rlaopt._typing import TensorDict
from rlaopt.expression.expression import AddExpression, Expression
from rlaopt.operator_split import OperatorSplit
from rlaopt.solvers.configs import SolverConfig
from rlaopt.solvers.proximal_gradient import prox_grad_func
from rlaopt.solvers.solver_base import OptimSolver


class ProxGradConfig(SolverConfig):
    """Configuration for the proximal gradient solver.

    Attributes:
        eta: Step size for the gradient update.
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence.
        use_acceleration: Whether to use acceleration techniques.
        use_linesearch: Whether to use line search for step size selection.
    """

    eta: float = Field(default=1.0, gt=0)
    max_iters: int = Field(default=5000, gt=0)
    tol: float = Field(default=1e-4, gt=0)
    use_acceleration: bool = False
    use_linesearch: bool = True


class ProximalGradient(OptimSolver):
    def __init__(
        self, config: ProxGradConfig, obj: Expression | AddExpression | OperatorSplit
    ):
        super().__init__(config)
        self._step = prox_grad_func._build_step(
            obj, config.use_acceleration, config.use_linesearch
        )

    def init_state(self, params: TensorDict) -> prox_grad_func.ProxGradState:
        return prox_grad_func._init_state(
            params, self.config.eta, self.config.use_acceleration
        )

    def step(
        self, params: TensorDict, state: prox_grad_func.ProxGradState
    ) -> Tuple[TensorDict, prox_grad_func.ProxGradState]:
        return self._step(params, state)

    def solve(
        self,
        obj: Expression | AddExpression | OperatorSplit,
        init_params: TensorDict = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return prox_grad_func.proximal_gradient(obj, self.config, init_params)
