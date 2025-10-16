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

    def __init__(self, config: SolverConfig):
        """Initialize the solver with an objective function.

        Args:
            config (SolverConfig): Configuration for the solver.
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
    def solve(
        self,
        problem: Expression | OperatorSplit,
        initial_params: TensorDict | None = None,
    ) -> TensorDict:
        """Solve the optimization problem defined by `problem`.

        Args:
            problem (OperatorSplit): The optimization problem to solve.
            initial_params (TensorDict): Initial parameters for the optimization.
            optim_state (OptimState): State of the optimizer.

        Returns:
            TensorDict: Optimized parameters after solving the problem.
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

    def __init__(self, config: LinSysSolverConfig):
        """Initialize the solver with configuration.

        Args:
            config: Configuration object for the solver.
        """
        self.config = config

    @abstractmethod
    def init_state(self, lin_sys) -> LinSysState:
        """Initialize the state of the solver.

        Args:
            lin_sys: The linear system to solve.

        Returns:
            Initial state for the solver containing iteration-specific variables.
        """
        pass

    @abstractmethod
    def step(self, lin_sys, state: LinSysState) -> LinSysState:
        """Perform a single iteration step of the solver.

        Args:
            lin_sys: The linear system to solve.
            state: Current state of the solver.

        Returns:
            Updated state after one iteration.
        """
        pass

    @abstractmethod
    def solve(self, lin_sys: LinSys) -> torch.Tensor:
        """Solve the linear system.

        This method performs iterative refinement until convergence criteria
        are met (either tolerance is achieved or max iterations reached).

        Args:
            lin_sys: The linear system to solve.

        Returns:
            Solution tensor w satisfying (A + reg*I)w = B.
        """
        pass
