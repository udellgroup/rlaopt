from typing import Callable, Union, Optional

import torch

from rlaopt.models.linops import LinOp
from rlaopt.solvers import PCG, SolverConfig, _is_solver_config
from rlaopt.utils import _is_str, _is_torch_tensor, Logger


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
        # TODO(pratik): turn these into spearate utility functions
        if not isinstance(A, (LinOp, torch.Tensor)):
            raise ValueError(
                f"A must be an instance of LinOp or a torch.Tensor. \
                             Received {type(A)}"
            )
        if not isinstance(b, torch.Tensor):
            raise ValueError(f"b must be a torch.Tensor. Received {type(b)}")
        if not isinstance(reg, float) or reg < 0:
            raise ValueError("reg must be a non-negative float")

    def _compute_internal_metrics(self, w: torch.Tensor):
        abs_res = torch.linalg.norm(self.b - (self.A @ w + self.reg * w))
        rel_res = abs_res / torch.linalg.norm(self.b)
        return {"abs_res": abs_res.item(), "rel_res": rel_res.item()}

    def _check_termination_criteria(
        self, internal_metrics: dict, atol: float, rtol: float
    ):
        abs_res = internal_metrics["abs_res"]
        return abs_res <= max(rtol * torch.linalg.norm(self.b), atol)

    def _get_logger_fn(
        self,
        callback_fn: Optional[Callable],
        callback_args: Optional[list],
        callback_kwargs: Optional[dict],
    ):
        if callback_fn is not None:

            def logger_fn(w):
                callback_log = callback_fn(w, self, *callback_args, **callback_kwargs)
                internal_metrics_log = self._compute_internal_metrics(w)
                return {
                    "callback": callback_log,
                    "internal_metrics": internal_metrics_log,
                }

        else:

            def logger_fn(w):
                internal_metrics_log = self._compute_internal_metrics(w)
                return {"internal_metrics": internal_metrics_log}

        return logger_fn

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

        _is_str(solver_name, "solver_name")
        if solver_name not in ["pcg"]:
            raise ValueError(f"Solver {solver_name} is not supported")
        _is_solver_config(solver_config, "solver_config")
        _is_torch_tensor(w_init, "w_init")

        log = {}
        # TODO make generic training loop that allows for early stopping
        if solver_name == "pcg":
            atol, rtol = solver_config.atol, solver_config.rtol

            def termination_fn(internal_metrics):
                return self._check_termination_criteria(internal_metrics, atol, rtol)

            logger_fn = self._get_logger_fn(callback_fn, callback_args, callback_kwargs)
            logger = Logger(log_freq=callback_freq)

            solver = PCG(
                self,
                w_init=w_init,
                device=solver_config.device,
                precond_config=solver_config.precond_config,
            )

            # Get initial log and check for termination
            log[0] = logger._compute_log(0, logger_fn, solver.w)
            if termination_fn(log[0]["metrics"]["internal_metrics"]):
                return solver.w, log

            # Training loop
            for i in range(1, solver_config.max_iters + 1):
                solver._step()
                log_i = logger._compute_log(i, logger_fn, solver.w)
                if log_i is not None:
                    log[i] = log_i
                    if termination_fn(log[i]["metrics"]["internal_metrics"]):
                        break

            return solver.w, log
