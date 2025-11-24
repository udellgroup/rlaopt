"""Base classes for optimization and linear system solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.linalg import LinSys
from rlaopt.solvers.configs_base import SolverConfig, StoppingCriteria


@dataclass(frozen=True)
class SolverState:
    """Base class for solver states.

    This class can be extended to include specific state variables
    required by different solvers.
    """

    iter_: int  # Current iteration count


@dataclass(frozen=True)
class SolverResult:
    """Base class for solver results.

    This class can be extended to include specific result variables
    returned by different solvers.
    """

    variable_values: TensorDict  # Optimized variable values


class OptimSolver(ABC):
    """Abstract base class for optimization solvers.

    This class defines the interface for all solvers in the library.
    Each solver must implement the `solve` method to perform optimization.
    """

    @abstractmethod
    def __init__(self, obj: Expression, config: SolverConfig):
        """Initialize the solver with an objective function.

        Args:
            obj (Expression): The objective function
                to optimize.
            config (SolverConfig): Configuration for the solver.
        """
        pass

    @abstractmethod
    def init_state(self, variable_values: TensorDict) -> SolverState:
        """Initialize the state of the optimizer.

        Args:
            variable_values (TensorDict): Initial variable values for the optimization.

        Returns:
            SolverState: Initial state of the optimizer.
        """
        pass

    @abstractmethod
    def step(
        self, variable_values: TensorDict, optim_state: SolverState
    ) -> tuple[TensorDict, SolverState]:
        """Performs a single optimization step.

        Args:
            variable_values (TensorDict): Current variable values.
            optim_state (SolverState): Current state of the optimizer.

        Returns:
            tuple[TensorDict, SolverState]: Updated variable values and optimizer state.
        """
        pass

    @abstractmethod
    def solve(
        self, variable_values: TensorDict, stopping_criteria: StoppingCriteria
    ) -> SolverResult:
        """Solve the optimization problem.

        Args:
            variable_values (TensorDict | None): Initial variable values.
                If None, the current variable values in the objective will be used.
            stopping_criteria (StoppingCriteria): Criteria to stop the optimization.

        Returns:
            SolverResult: Result of the optimization containing optimized variable
                values among other metrics.
        """
        pass


class LinSysSolver(ABC):
    """Abstract base class for linear system solvers.

    This class defines the interface for all linear system solvers in the library.
    Each solver must implement methods to initialize state, perform iteration steps,
    and solve linear systems of the form AW = B.

    Solvers are iterative methods (e.g., Conjugate Gradient)
    that progressively refine a solution until convergence criteria are met.
    """

    @abstractmethod
    def __init__(self, lin_sys: LinSys, config: SolverConfig):
        """Initialize the solver.

        Args:
            lin_sys (LinSys): The linear system to solve.
            config (SolverConfig): Configuration object for the solver.
        """
        pass

    @abstractmethod
    def init_state(self, params: torch.Tensor) -> SolverState:
        """Initialize the state of the solver.

        Args:
            params: Initial parameters (solution estimate).

        Returns:
            SolverState: Initial state for the solver containing
                iteration-specific variables.
        """
        pass

    @abstractmethod
    def step(
        self, params: torch.Tensor, state: SolverState
    ) -> tuple[torch.Tensor, SolverState]:
        """Perform a single iteration step of the solver.

        Args:
            params: Current parameters (solution estimate).
            state: Current state of the solver.

        Returns:
            tuple[torch.Tensor, SolverState]: Updated parameters and state
                after one iteration.
        """
        pass

    @abstractmethod
    def solve(
        self, params: torch.Tensor, stopping_criteria: StoppingCriteria
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the linear system AW = B.

        Args:
            params: Initial parameters (solution estimate).
            stopping_criteria (StoppingCriteria): Criteria to stop the solver.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Optimized parameters and
                final residual norm.
        """
        pass
