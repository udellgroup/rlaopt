from abc import ABC, abstractmethod
from typing import Optional, Callable

import torch

from rlaopt.solvers import SolverConfig, Solver
from rlaopt.utils import Logger


class Model(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def _check_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def _compute_internal_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def _check_termination_criteria(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_logger_fn(self, *args, **kwargs):
        pass

    def _train(
        self,
        logger: Logger,
        logger_fn: Callable,
        termination_fn: Callable,
        solver: Solver,
        max_iters: int,
    ):
        log = {}

        # Get initial log and check for termination
        log[0] = logger._compute_log(0, logger_fn, solver.w)
        if termination_fn(log[0]["metrics"]["internal_metrics"]):
            return solver.w, log

        # Training loop
        for i in range(1, max_iters + 1):
            solver._step()
            log_i = logger._compute_log(i, logger_fn, solver.w)
            if log_i is not None:
                log[i] = log_i
                if termination_fn(log[i]["metrics"]["internal_metrics"]):
                    break

        return solver.w, log

    @abstractmethod
    def solve(
        self,
        solver_name: str,
        solver_config: SolverConfig,
        w_init: torch.Tensor,
        callback_fn: Optional[Callable] = None,
        callback_args: Optional[list] = [],
        callback_kwargs: Optional[dict] = {},
        callback_freq: Optional[int] = 10,
        log_in_wandb: Optional[bool] = False,
        wandb_init_kwargs: Optional[dict] = None,
    ):
        pass
