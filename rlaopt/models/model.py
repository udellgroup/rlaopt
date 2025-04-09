from abc import ABC, abstractmethod
from typing import Optional, Callable
from warnings import warn

import torch

from rlaopt.solvers import SolverConfig, Solver
from rlaopt.utils import Logger


__all__ = ["Model"]


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

    def _get_wandb_kwargs(
        self,
        log_in_wandb: bool,
        wandb_init_kwargs: dict,
        solver_name: str,
        solver_config: SolverConfig,
        callback_freq: int,
    ):
        if log_in_wandb:
            wandb_kwargs = {
                "config": {
                    "solver_name": solver_name,
                    "solver_config": solver_config.to_dict(),
                    "callback_freq": callback_freq,
                },
            }

            # Ensure wandb_init_kwargs is merged into wandb_kwargs
            if wandb_init_kwargs is not None:
                for key, value in wandb_init_kwargs.items():
                    if key == "config":
                        warn(
                            "Found 'config' key in wandb_init_kwargs. "
                            "Merging with internally specified 'config' key."
                        )

                        # Merge the config dictionary
                        wandb_kwargs["config"].update(value)
                    else:
                        wandb_kwargs[key] = value
        else:
            wandb_kwargs = None

        return wandb_kwargs

    def _train(
        self,
        logger: Logger,
        termination_fn: Callable,
        solver: Solver,
        max_iters: int,
    ):
        log = {}

        # Get initial log and check for termination
        log[0] = logger._compute_log(0, solver.W)
        if termination_fn(log[0]["metrics"]["internal_metrics"]):
            return solver.W, log

        # Training loop
        for i in range(1, max_iters + 1):
            solver._step()
            log_i = logger._compute_log(i, solver.W)
            if log_i is not None:
                log[i] = log_i
                if termination_fn(log[i]["metrics"]["internal_metrics"]):
                    break

        logger._terminate()

        return solver.W, log

    @abstractmethod
    def solve(
        self,
        solver_config: SolverConfig,
        W_init: torch.Tensor,
        callback_fn: Optional[Callable] = None,
        callback_args: Optional[list] = [],
        callback_kwargs: Optional[dict] = {},
        callback_freq: Optional[int] = 10,
        log_in_wandb: Optional[bool] = False,
        wandb_init_kwargs: Optional[dict] = None,
    ):
        pass
