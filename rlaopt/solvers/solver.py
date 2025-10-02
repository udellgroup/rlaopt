from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .configs import SolverConfig
from ..expression.expression import Expression
from ..operator_split import OperatorSplit
from .._typing import OptimState, TensorDict

class Solver(ABC):
    """
    Abstract base class for optimization solvers.

    This class defines the interface for all solvers in the library.
    Each solver must implement the `solve` method to perform optimization.
    """

    def __init__(self, config: SolverConfig):
        """
        Initialize the solver with an objective function.

        Args:
            config (SolverConfig): Configuration for the solver.
        """
        self.config = config
    
    @abstractmethod
    def init_state(self, params: TensorDict) -> OptimState:
        """
        Initialize the state of the optimizer.

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
        initial_params: Optional[TensorDict] = None, 
    ) -> TensorDict:
        """
        Solve the optimization problem defined by `problem`.

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
        self, 
        params: TensorDict, 
        optim_state: OptimState
    ) -> Tuple[TensorDict, OptimState]:
        """
        Performs a single optimization step.

        Args:
            params (TensorDict): Current parameters.
            optim_state (OptimState): Current state of the optimizer.

        Returns:
            Tuple[TensorDict, OptimState]: Updated parameters and optimizer state.
        """
        pass