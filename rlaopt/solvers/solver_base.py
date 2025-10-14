from abc import ABC, abstractmethod

from rlaopt._typing import OptimState, TensorDict
from rlaopt.expression.expression import Expression
from rlaopt.operator_split import OperatorSplit
from rlaopt.solvers.configs import SolverConfig


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
