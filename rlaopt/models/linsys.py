from typing import Callable, Union, Optional

import torch

from rlaopt.solvers import PCG, SolverConfig
from rlaopt.models.linops import LinOp


class LinSys:
    def __init__(
        self, A: Union[LinOp, torch.Tensor], b: torch.Tensor, reg: Optional[float] = 0.0
    ):
        self._check_inputs(A, b, reg)
        self._A = A
        self._b = b
        self._reg = reg

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def reg(self):
        return self._reg

    def _check_inputs(self, A: Union[LinOp, torch.Tensor], b: torch.Tensor, reg: float):
        if not isinstance(A, (LinOp, torch.Tensor)):
            raise ValueError(
                f"A must be an instance of LinOp or a torch.Tensor. \
                             Received {type(A)}"
            )
        if not isinstance(b, torch.Tensor):
            raise ValueError(f"b must be a torch.Tensor. Received {type(b)}")
        if not isinstance(reg, float) or reg < 0:
            raise ValueError("reg must be a non-negative float")

    def solve(
        self,
        solver_name: str,
        solver_config: SolverConfig,
        w_init: torch.Tensor,
        callback_fn: Optional[Callable] = None,
        callback_args: Optional[list] = [],
        callback_kwargs: Optional[dict] = {},
        callback_freq: Optional[int] = 10,
    ):
        log = {}

        if solver_name not in ["pcg"]:
            raise ValueError(f"Solver {solver_name} is not supported")
        # To Do make generic training loop that allows for early stopping
        if solver_name == "pcg":
            solver = PCG(
                self, w_init=w_init, precond_config=solver_config.precond_config
            )
            atol, rtol = solver_config.atol, solver_config.rtol
            for i in range(solver_config.max_iters):
                solver._step()
                if i % callback_freq == 0:
                    abs_res = torch.linalg.norm(
                        self.b - (self.A @ solver.w + self.reg * solver.w)
                    )
                    rel_res = abs_res / torch.linalg.norm(self.b)
                    log[i] = {"abs_res": abs_res, "rel_res": rel_res}
                    if callback_fn is not None:
                        callback_log = callback_fn(
                            solver.w, self, *callback_args, **callback_kwargs
                        )
                        log[i]["callback"] = callback_log
                    if abs_res <= max(rtol * torch.linalg.norm(self.b), atol):
                        break
            return solver.w, log
