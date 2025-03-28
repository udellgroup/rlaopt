from typing import Any, Callable, Union, Optional

import torch

from .model import Model
from rlaopt.solvers import _is_solver_config, _get_solver_name, _get_solver
from rlaopt.utils import _is_callable, _is_torch_tensor, _is_nonneg_float, Logger
from rlaopt.linops import LinOpType, _is_linop_or_torch_tensor


__all__ = ["LinSys"]


class LinSys(Model):
    """Model for solving positive-definite linear systems (A + reg * I)w = b."""

    def __init__(
        self,
        A: Union[LinOpType, torch.Tensor],
        B: torch.Tensor,
        reg: Optional[float] = 0.0,
        A_row_oracle: Optional[Callable] = None,
        A_blk_oracle: Optional[Callable] = None,
    ):
        """Initialize LinSys model.

        Args:
            A (Union[LinOpType, torch.Tensor]): Linear operator or matrix A.
            b (torch.Tensor): Right-hand side b.
            reg (Optional[float], optional): Regularization parameter. Defaults to 0.0.
            A_row_oracle (Optional[Callable], optional): Oracle for row-wise operations.
              Defaults to None.
            A_blk_oracle (Optional[Callable], optional): Oracle for block-wise
            operations. Defaults to None.
        """
        self._check_inputs(A, B, reg, A_row_oracle, A_blk_oracle)
        self._A = A
        self._B = B
        if self._B.ndim == 1:
            self._B = self._B.unsqueeze(-1)
        self._reg = reg
        self._A_row_oracle = A_row_oracle
        self._A_blk_oracle = A_blk_oracle

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def reg(self):
        return self._reg

    @property
    def A_row_oracle(self):
        return self._A_row_oracle

    @property
    def A_blk_oracle(self):
        return self._A_blk_oracle

    def _check_inputs(
        self,
        A: Any,
        B: Any,
        reg: Any,
        A_row_oracle: Optional[Any],
        A_blk_oracle: Optional[Any],
    ):
        _is_linop_or_torch_tensor(A, "A")
        _is_torch_tensor(B, "B")
        _is_nonneg_float(reg, "reg")
        if A_row_oracle is not None:
            _is_callable(A_row_oracle, "A_row_oracle")
        if A_blk_oracle is not None:
            _is_callable(A_blk_oracle, "A_blk_oracle")

        # If one of the oracles is provided, the other one must also be provided
        if A_row_oracle is not None and A_blk_oracle is None:
            raise ValueError(
                "A_blk_oracle must be provided if A_row_oracle is provided"
            )
        if A_blk_oracle is not None and A_row_oracle is None:
            raise ValueError(
                "A_row_oracle must be provided if A_blk_oracle is provided"
            )

    def _compute_internal_metrics(self, W: torch.Tensor):
        abs_res = torch.linalg.norm(self.B - (self.A @ W + self.reg * W), dim=0, ord=2)
        rel_res = abs_res / torch.linalg.norm(self.B, dim=0, ord=2)
        return {"abs_res": abs_res, "rel_res": rel_res}

    def _check_termination_criteria(
        self, internal_metrics: dict, atol: float, rtol: float
    ):
        abs_res = internal_metrics["abs_res"]
        comp_tol = torch.clamp(rtol * torch.linalg.norm(self.B, dim=0, ord=2), min=atol)
        return (abs_res <= comp_tol).all().item()

    def solve(
        self,
        solver_config,
        W_init,
        callback_fn=None,
        callback_args=[],
        callback_kwargs={},
        callback_freq=10,
        log_in_wandb=False,
        wandb_init_kwargs=None,
    ):
        _is_solver_config(solver_config, "solver_config")
        _is_torch_tensor(W_init, "W_init")
        if log_in_wandb and wandb_init_kwargs is None:
            raise ValueError(
                "wandb_init_kwargs must be specified if log_in_wandb is True"
            )

        # Termination criteria
        atol, rtol = solver_config.atol, solver_config.rtol

        def termination_fn(internal_metrics):
            return self._check_termination_criteria(internal_metrics, atol, rtol)

        # Setup logging
        log_fn = self._get_log_fn(callback_fn, callback_args, callback_kwargs)
        wandb_kwargs = self._get_wandb_kwargs(
            log_in_wandb=log_in_wandb,
            wandb_init_kwargs=wandb_init_kwargs,
            solver_name=_get_solver_name(solver_config),
            solver_config=solver_config,
            callback_freq=callback_freq,
        )
        logger = Logger(
            log_freq=callback_freq,
            log_fn=log_fn,
            wandb_kwargs=wandb_kwargs,
        )

        # Get solver
        solver = _get_solver(model=self, W_init=W_init, solver_config=solver_config)

        # Run solver
        solution, log = self._train(
            logger=logger,
            termination_fn=termination_fn,
            solver=solver,
            max_iters=solver_config.max_iters,
        )

        return solution, log
