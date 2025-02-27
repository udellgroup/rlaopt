from typing import Callable, Union, Optional

import torch

from rlaopt.models.model import Model
from rlaopt.models.linops import LinOp
from rlaopt.solvers import PCG, _is_solver_config
from rlaopt.utils import _is_str, _is_torch_tensor, Logger


class LinSys(Model):
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
        # TODO(pratik): turn these into separate utility functions
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

    def _get_log_fn(
        self,
        callback_fn: Optional[Callable],
        callback_args: Optional[list],
        callback_kwargs: Optional[dict],
    ):
        if callback_fn is not None:

            def log_fn(w):
                callback_log = callback_fn(w, self, *callback_args, **callback_kwargs)
                internal_metrics_log = self._compute_internal_metrics(w)
                return {
                    "callback": callback_log,
                    "internal_metrics": internal_metrics_log,
                }

        else:

            def log_fn(w):
                internal_metrics_log = self._compute_internal_metrics(w)
                return {"internal_metrics": internal_metrics_log}

        return log_fn

    def solve(
        self,
        solver_name,
        solver_config,
        w_init,
        callback_fn: Optional[Callable] = None,
        callback_args: Optional[list] = [],
        callback_kwargs: Optional[dict] = {},
        callback_freq: Optional[int] = 10,
        log_in_wandb: Optional[bool] = False,
        wandb_init_kwargs: Optional[dict] = None,
    ):

        _is_str(solver_name, "solver_name")
        if solver_name not in ["pcg"]:
            raise ValueError(f"Solver {solver_name} is not supported")
        _is_solver_config(solver_config, "solver_config")
        _is_torch_tensor(w_init, "w_init")
        if log_in_wandb and wandb_init_kwargs is None:
            raise ValueError(
                "wandb_init_kwargs must be specified if log_in_wandb is True"
            )

        # TODO(pratik): make generic training loop
        if solver_name == "pcg":
            atol, rtol = solver_config.atol, solver_config.rtol

            def termination_fn(internal_metrics):
                return self._check_termination_criteria(internal_metrics, atol, rtol)

            log_fn = self._get_log_fn(callback_fn, callback_args, callback_kwargs)

            wandb_kwargs = self._get_wandb_kwargs(
                log_in_wandb=log_in_wandb,
                wandb_init_kwargs=wandb_init_kwargs,
                solver_name=solver_name,
                solver_config=solver_config,
                callback_freq=callback_freq,
            )

            logger = Logger(
                log_freq=callback_freq,
                log_fn=log_fn,
                wandb_kwargs=wandb_kwargs,
            )

            solver = PCG(
                self,
                w_init=w_init,
                device=solver_config.device,
                precond_config=solver_config.precond_config,
            )

            solution, log = self._train(
                logger=logger,
                termination_fn=termination_fn,
                solver=solver,
                max_iters=solver_config.max_iters,
            )

            return solution, log
