"""Gradient oracles and optimization utilities for first-order methods.

This module provides the core gradient computation strategies (full, SGD, SVRG, SAGA),
gradient transformations (identity, preconditioning), proximal updates, and
error metrics used by all gradient based solvers.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from functools import partial
from typing import Any, Callable

import torch
from torch.nn.modules.loss import _Loss

from rlaopt.data import DataLoader
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.sapphire_split import SapphireSplit

from .gradient_solver_states import (
    GradSolverState,
    ProxGradState,
    SAGAState,
    SapphireState,
    SGDState,
    SVRGState,
)

##### Gradient Oracles #####


class StochasticGradientOracle(ABC):
    """Abstract base class for stochastic gradient computation strategies.

    Subclasses must implement the gradient method to define how stochastic
    gradients are computed for use in variance-reduced or mini-batch methods.
    """

    @classmethod
    def build_gradient_fn(
        cls, **kwargs: Any
    ) -> Callable[
        [TensorDict, SapphireState, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        tuple[TensorDict, SapphireState],
    ]:
        """Build a gradient function with pre-bound parameters.

        Args:
            **kwargs: Additional keyword arguments to pass to the gradient method.

        Returns:
            Partial function that computes gradients given var_vals, state, and batch.
        """
        return partial(cls.gradient, **kwargs)

    @staticmethod
    @abstractmethod
    def gradient(
        var_vals: TensorDict,
        state: SapphireState,
        loader: DataLoader,
        batch_gradient_fn: Callable[
            [TensorDict, torch.Tensor, torch.Tensor], TensorDict
        ],
        **kwargs: Any,
    ) -> TensorDict:
        """Compute stochastic gradient estimate.

        Args:
            var_vals: Current variable values.
            state: Current optimizer state.
            loader: DataLoader providing mini-batches via ``get_batch()``.
            batch_gradient_fn: Function computing the per-batch gradient
                given (var_vals, X_batch, y_batch).
            **kwargs: Additional method-specific parameters.

        Returns:
            Stochastic gradient estimate as a TensorDict.
        """
        pass


class SGDOracle(StochasticGradientOracle):
    """Stochastic Gradient Descent oracle using mini-batch gradients."""

    @staticmethod
    def gradient(
        var_vals: TensorDict,
        state: SGDState,
        loader: DataLoader,
        batch_gradient_fn: Callable[
            [TensorDict, torch.Tensor, torch.Tensor], TensorDict
        ],
    ):
        """Compute standard SGD mini-batch gradient.

        Args:
            var_vals: Current variable values.
            state: SGD optimizer state.
            loader: DataLoader providing mini-batches via ``get_batch()``.
            batch_gradient_fn: Function to compute batch gradients.

        Returns:
            Tuple of (gradient, unchanged_state).
        """
        X_batch, y_batch, _ = loader.get_batch()
        return batch_gradient_fn(var_vals, X_batch, y_batch), state


class SVRGOracle(StochasticGradientOracle):
    """Stochastic Variance Reduced Gradient (SVRG) oracle.

    SVRG reduces gradient variance by maintaining a snapshot point and its
    full gradient, using the difference between current and snapshot batch
    gradients as a control variate.
    """

    @staticmethod
    def gradient(
        var_vals: TensorDict,
        state: SVRGState,
        loader: DataLoader,
        batch_gradient_fn: Callable[
            [TensorDict, torch.Tensor, torch.Tensor], TensorDict
        ],
        full_gradient_fn: Callable[[TensorDict], TensorDict],
        update_threshold: int,
    ):
        """Compute SVRG variance-reduced gradient estimate.

        Args:
            var_vals: Current variable values.
            state: SVRG optimizer state containing snapshot and snapshot gradient.
            loader: DataLoader providing mini-batches via ``get_batch()``.
            batch_gradient_fn: Function to compute batch gradients.
            full_gradient_fn: Function to compute full dataset gradient.
            update_threshold: Number of iterations between snapshot updates.

        Returns:
            Tuple of (SVRG gradient estimate, updated state).
        """
        X_batch, y_batch, _ = loader.get_batch()
        # Update snapshot if needed
        if state.iter_ % update_threshold == 0:
            grad_snapshot = full_gradient_fn(var_vals)
            state = replace(state, snapshot=var_vals, snapshot_grad=grad_snapshot)

        # Compute SVRG gradient
        v = (
            batch_gradient_fn(var_vals, X_batch, y_batch)
            - batch_gradient_fn(state.snapshot, X_batch, y_batch)
            + state.snapshot_grad
        )
        return v, state


class SAGAOracle(StochasticGradientOracle):
    """Stochastic Average Gradient Augmented (SAGA) oracle.

    SAGA maintains a table of individual gradient components and computes
    variance-reduced gradients using the difference between current and
    stored gradients for sampled data points.
    """

    @staticmethod
    def gradient(
        var_vals: TensorDict,
        state: SAGAState,
        loader: DataLoader,
        loss_fn: _Loss,
        prediction_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        grad_reg: Callable[[TensorDict], TensorDict | float],
        n: int,
        has_intercept: bool,
    ):
        """Compute SAGA variance-reduced gradient estimate.

        Args:
            var_vals: Current variable values.
            state: SAGA optimizer state containing gradient table and average.
            loader: DataLoader providing mini-batches via ``get_batch()``.
            loss_fn: Loss function for computing gradients.
            prediction_fn: Function to compute model predictions.
            grad_reg: Function returning the regularizer gradient for the
                given var_vals.
            n: Total number of samples in the dataset.
            has_intercept: Whether the model includes an intercept term.

        Returns:
            Tuple of (SAGA gradient estimate, updated state with new table/average).
        """
        #  Extract data tensors and batch indices
        X_batch, y_batch, batch_indices = loader.get_batch()
        batch_size = X_batch.shape[0]

        # Get table weights for current batch
        tbl_weights = state.table[batch_indices]

        # Compute prediction on batch
        if has_intercept:
            intercept_tensor = var_vals["intercept"]
        else:
            intercept_tensor = None

        y_pre = prediction_fn(var_vals["beta"], intercept_tensor, X_batch)

        # Compute new table weights
        loss_fn.reduction = "sum"
        new_weights = torch.func.grad(loss_fn)(y_pre, y_batch)
        loss_fn.reduction = "mean"

        # Get aux tensor for beta coefficients
        aux_beta = X_batch.T @ (new_weights - tbl_weights)

        # Get aux tensor for intercept (if present)
        if has_intercept:
            aux_intercept = (new_weights - tbl_weights).sum()
            aux = TensorDict({"beta": aux_beta, "intercept": aux_intercept})
        else:
            aux = TensorDict({"beta": aux_beta})

        # Regularizer gradient
        g_reg = grad_reg(var_vals)

        # Get new gradient
        v = state.grad_avg + 1 / batch_size * aux + g_reg

        # Update the table and average
        grad_avg_new = state.grad_avg + 1 / n * aux

        table_new = state.table.clone()
        table_new[batch_indices] = new_weights

        return v, replace(state, table=table_new, grad_avg=grad_avg_new)


##### Transform functions #####
def identity_transform(grads: TensorDict, state: SapphireState):
    """Apply identity transformation to gradients (no-op).

    Args:
        grads: Gradient TensorDict.
        state: Current SAPPHIRE state.

    Returns:
        Unchanged gradients.
    """
    return grads


def precond_grad(grads: TensorDict, state: SapphireState) -> TensorDict:
    """Apply preconditioner to gradients.

    Computes P^{-1} @ grads where P is the preconditioner stored in state.

    Args:
        grads: Gradient TensorDict to precondition.
        state: SAPPHIRE state containing preconditioner P.

    Returns:
        Preconditioned gradients as a TensorDict.
    """
    return grads.from_flat_tensor(state.P.inv @ grads.to_flat_tensor())


##### Update functions #####
def nest_accel_update(
    var_vals: TensorDict, state: ProxGradState
) -> tuple[TensorDict, ProxGradState]:
    """Compute the accelerated (momentum) iterate.

    Applies Nesterov momentum with adaptive momentum coefficient based on
    iteration count: momentum_scale = k / (k + 3).

    Args:
        var_vals: Current variable values.
        state: Proximal gradient state containing previous iterate.

    Returns:
        Tuple of (accelerated iterate, updated state with saved previous values).
    """
    var_vals_prev = var_vals.clone()
    momentum_scale = state.iter_ / (state.iter_ + 3)
    return var_vals + momentum_scale * (var_vals - state.variable_values_prev), replace(
        state, variable_values_prev=var_vals_prev
    )


def prox_update(
    var_vals: TensorDict,
    updates: TensorDict,
    state: GradSolverState,
    prox_fn: Callable[[TensorDict, float], TensorDict],
):
    """Apply proximal update step.

    Computes prox_g(x - eta * updates), where g is the non-smooth part of the
    objective and eta is the step size.

    Args:
        var_vals: Current variable values.
        updates: Update direction (typically gradients).
        state: Proximal gradient state containing step size eta.
        prox_fn: Proximal operator function.

    Returns:
        Tuple of (updated variables, unchanged state).
    """
    return prox_fn(var_vals - state.eta * updates, state.eta), state


def prox_update_P(
    var_vals: TensorDict,
    grads: TensorDict,
    state: SapphireState,
    subproblem_iters: int,
    prox_fn: Callable[[TensorDict, float], TensorDict],
) -> tuple[TensorDict, SapphireState]:
    """Compute scaled proximal operator in SAPPHIRE via APG.

    Given preconditioner P stored in the SAPPHIRE state, this function
    approximately evaluates the scaled proximal mapping:
    prox_P(x_k - eta * grads), using Accelerated Proximal Gradient.

    Args:
        var_vals: Current variable values.
        grads: Gradient direction.
        state: SAPPHIRE state containing preconditioner P, step size eta, and norm.
        subproblem_iters: Number of APG iterations for subproblem.
        prox_fn: Proximal operator function.

    Returns:
        Tuple of (approximate solution to scaled proximal subproblem, unchanged state).
    """
    alpha = 1 / state.P.norm
    t_old = 1.0
    z0 = var_vals.clone()
    y = z0
    z_old = z0

    for _ in range(subproblem_iters):
        Pd = y.from_flat_tensor(state.P @ (y - var_vals).to_flat_tensor())
        y_int = y - alpha * (state.eta * grads + Pd)

        z0 = prox_fn(y_int, state.eta * alpha)
        t = (1 + (1 + 4 * t_old**2) ** (0.5)) / 2
        y = z0 + (t_old - 1) / t * (z0 - z_old)

        t_old = t
        z_old = z0

    return y, state


def linesearch(
    grads: TensorDict,
    var_vals: TensorDict,
    state: ProxGradState,
    f: Callable[[TensorDict], torch.Tensor],
    apply_updates: Callable[[TensorDict, ProxGradState], TensorDict],
) -> tuple[TensorDict, ProxGradState]:
    """Perform backtracking line search to find appropriate step size.

    Backtracking line search with reduction factor beta=0.5, ensuring sufficient
    decrease condition based on the quadratic upper bound at the proximal point.

    Args:
        grads: Gradient TensorDict used to form candidate updates.
        var_vals: Current variable values.
        state: Proximal gradient state containing initial step size eta.
        f: Objective function to minimize.
        apply_updates: Callable applying the update step (e.g. proximal
            map) to a candidate iterate.

    Returns:
        Tuple of (accepted iterate, updated state with new step size and error).
    """
    beta = 0.5
    f0 = f(var_vals)
    cond = False

    def linesearch_step(var_vals: TensorDict, state: ProxGradState):
        z = apply_updates(var_vals - state.eta * grads, state.eta)
        d = z - var_vals
        u = f0 + grads.flat_dot(d) + 1 / (2 * state.eta) * (d.flat_norm() ** 2)

        if f(z) <= u:
            err_new = d.flat_norm() / state.eta
            # Simple forward tracking to ensure step_size does not become too small:
            eta_new = state.eta / beta
            return True, z, replace(state, err=err_new, eta=eta_new)
        else:
            eta_new = beta * state.eta
            return False, var_vals, replace(state, eta=eta_new)

    while not cond:
        cond, var_vals, state = linesearch_step(var_vals, state)

    return var_vals, state


def load_batch(op_split: SapphireSplit):
    """Load a mini-batch from the dataloader.

    Args:
        op_split: SAPPHIRE operator splitting object containing dataloader.

    Returns:
        Mini-batch from the dataloader.
    """
    return op_split.loader.get_batch()


#### Error metrics ####
def grad_mapping_norm(
    var_vals: TensorDict,
    state: GradSolverState,
    full_gradient_fn: Callable[[TensorDict], TensorDict],
    prox_fn: Callable[[TensorDict, float], TensorDict],
) -> tuple[TensorDict, GradSolverState]:
    """Compute gradient mapping norm as convergence metric.

    The gradient mapping is defined as:
    G(x) = (x - prox(x - eta * g(x), eta)) / eta,
    where g(x) is the gradient evaluated at x, and eta is the stepsize.
    For composite objectives ||G(x)|| measures proximity to a stationary point.
    When the composite objective is also convex, ||G(x)|| = 0 means x is a minimizer.
    In the smooth case, G(x) collapses to the gradient.

    Args:
        var_vals: Current variable values.
        state: Gradient solver state containing step size eta.
        full_gradient_fn: Function to compute full dataset gradient.
        prox_fn: Proximal operator function.

    Returns:
        Tuple of (unchanged var_vals, updated state with error metric).
    """
    grads = full_gradient_fn(var_vals)
    var_vals_new, _ = prox_update(var_vals, grads, state, prox_fn)
    G = (var_vals - var_vals_new) / state.eta
    return var_vals, replace(state, err=G.flat_norm().item())
