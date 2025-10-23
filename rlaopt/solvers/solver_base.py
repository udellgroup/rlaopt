"""Base classes for optimization and linear system solvers."""

from abc import ABC, abstractmethod

import torch

from rlaopt._typing import LinSysState, OptimState, TensorDict
from rlaopt.expression.expression import Expression
from rlaopt.linalg import LinSys
from rlaopt.operator_split import OperatorSplit
from rlaopt.solvers.configs_base import LinSysSolverConfig, SolverConfig


class OptimSolver(ABC):
    """Abstract base class for optimization solvers.

    This class defines the interface for all solvers in the library.
    Each solver must implement the `solve` method to perform optimization.
    """

    def __init__(self, config: SolverConfig, obj: Expression | OperatorSplit):
        """Initialize the solver with an objective function.

        Args:
            config (SolverConfig): Configuration for the solver.
            obj (Expression | AddExpression | OperatorSplit): The objective function
                to optimize.
        """
        self.config = config

    @abstractmethod
    def init_state(self, params: TensorDict) -> OptimState:
        """Initialize the state of the optimizer.

        Args:
            params (TensorDict): Initial parameters for the optimization.

        Returns:
            OptimState: Initial state of the optimizer.
        """
        pass

    @abstractmethod
    def step(
        self, params: TensorDict, optim_state: OptimState
    ) -> tuple[TensorDict, OptimState]:
        """Performs a single optimization step.

        Args:
            params (TensorDict): Current parameters.
            optim_state (OptimState): Current state of the optimizer.

        Returns:
            tuple[TensorDict, OptimState]: Updated parameters and optimizer state.
        """
        pass


class LinSysSolver(ABC):
    """Abstract base class for linear system solvers.

    This class defines the interface for all linear system solvers in the library.
    Each solver must implement methods to initialize state, perform iteration steps,
    and solve linear systems of the form (A + reg*I)w = B.

    Solvers are iterative methods (e.g., Conjugate Gradient)
    that progressively refine a solution until convergence criteria are met.
    """

    @abstractmethod
    def __init__(self, config: LinSysSolverConfig, lin_sys: LinSys):
        """Initialize the solver.

        Args:
            config: Configuration object for the solver.
            lin_sys: The linear system to solve.
        """
        pass

    @abstractmethod
    def init_state(self, params: torch.Tensor) -> LinSysState:
        """Initialize the state of the solver.

        Args:
            params: Initial parameters (solution estimate).

        Returns:
            Initial state for the solver containing iteration-specific variables.
        """
        pass

    @abstractmethod
    def step(
        self, params: torch.Tensor, state: LinSysState
    ) -> tuple[torch.Tensor, LinSysState]:
        """Perform a single iteration step of the solver.

        Args:
            params: Current parameters (solution estimate).
            state: Current state of the solver.

        Returns:
            Updated state after one iteration.
        """
        pass
