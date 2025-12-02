from dataclasses import dataclass
from time import perf_counter

from pydantic import Field

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.solvers.solver_base import ConvergenceStatus, OptimSolver, OptimResult
from rlaopt.solvers.configs_base import StoppingCriteria
from rlaopt.splitting import ProxGradSplit, SapphireSplit

from .optim_configs import GradSolverConfig, ProxGradConfig, SapphireConfig
from .optim_states import get_solver_state, GradSolverState, ProxGradState, SapphireState
from .step_builder import get_step_fn


class GradSolverStoppingCriteria(StoppingCriteria):
    """Stopping criteria for Gradient-based solvers.

    Attributes:
        max_iters: Maximum number of iterations.
        tol: Tolerance for convergence based on the error metric.
    """

    eps_abs: float = Field(default=1e-4, gt=0.0)
    eps_rel: float = Field(default=1e-4, gt=0.0)

@dataclass(frozen=True)
class GradSolverResult(OptimResult):
    """Result container for the proximal gradient solver.

    Attributes:
        variable_values: Optimized variable values.
        convergence_status: Status indicating how the solver terminated.
        num_iters: Number of iterations performed.
        solver_time: Time taken by the solver in seconds.
        err: Final error metric upon termination.
    """

    err: float

class BaseGradientSolver(OptimSolver):
    def __init__(
            self, 
            obj: Expression, 
            config: GradSolverConfig, 
            target_config: GradSolverConfig,
            detach = True
    ):
        if not isinstance(obj, Expression):
            raise ValueError(f"{type(self).__name__} solver requires an Expression objective.")
        
        if not isinstance(config, target_config):
            raise ValueError(f"{type(self).__name__} solver requires an {target_config} config.")

        super().__init__(obj, config, detach)
        self._config = config
        self._detach = detach
        self._op_split = _split_objective(obj, config)
        self._step = get_step_fn(config, self._op_split)
    
    
    def init_state(self, variable_values: TensorDict)-> GradSolverState:
        """Initialize the solver state.

        Args:
            variable_values: Initial variable values.

        Returns:
            Initial solver state.
        """
        return get_solver_state(self._op_split, variable_values, self._config)
    
    
    def solve(
            self, 
            variable_values: TensorDict | None = None, 
            stopping_criteria: GradSolverStoppingCriteria = GradSolverStoppingCriteria()
    )->GradSolverResult:  
        ts = perf_counter()
        
        if variable_values is None:
            variable_values = self._op_split.variable_values
        
        state = self.init_state(variable_values)
        max_iters = stopping_criteria.max_iters
        eps_abs, eps_rel = stopping_criteria.eps_abs, stopping_criteria.eps_rel

        while state.err > eps_abs + eps_rel * variable_values.flat_norm() and state.iter_ < max_iters:
            variable_values, state = self.step(variable_values, state)
        
        if state.err <= eps_abs + eps_rel * variable_values.flat_norm() :
            convergence_status = ConvergenceStatus.CONVERGED
        else:
            convergence_status = ConvergenceStatus.NOT_CONVERGED
        
        tf = perf_counter() - ts

        return GradSolverResult(
            variable_values, convergence_status, state.iter_, solver_time=tf, err=state.err
        )
    
    def step(self, variable_values:TensorDict, state: GradSolverState):
        """Perform a single optimization step.

        Args:
            variable_values: Current variable values.
            state: Current solver state.

        Returns:
            Tuple of updated variable values and state.
        """
        variable_values, state = self._step(variable_values, state)
        if self._detach:
            variable_values = variable_values.detach()
        return variable_values, state


class ProxGrad(BaseGradientSolver):
    """Proximal gradient solver for optimization problems.

    Solves problems of the form:
        minimize f(x) + g(x)
    where f is smooth (differentiable) and g is proxable (has an efficient
    proximal operator).

    Supports multiple variants:
    - Basic proximal gradient (fixed step size)
    - Accelerated proximal gradient (Nesterov momentum)
    - Backtracking line search for adaptive step sizes
    - Combinations of acceleration and line search
    """
    def __init__(self, obj: Expression, config: ProxGradConfig, detach=True):
        """Initialize the proximal gradient solver.

        Args:
            obj: The optimization objective (Expression).
            config: Configuration for the solver.
        """
        if not isinstance(config, ProxGradConfig):
            raise ValueError("ProxGrad solver requires a ProxGradConfig configuration.")
        super().__init__(obj, config, ProxGradConfig, detach)
    
    def init_state(self, variable_values: TensorDict)->ProxGradState:
        return super().init_state(variable_values)
    
    def step(
            self, variable_values: TensorDict, state: ProxGradState
    )->tuple[TensorDict, ProxGradState]:
        return super().step(variable_values, state)


class Sapphire(BaseGradientSolver):
    def __init__(self, obj: Expression, config: SapphireConfig, detach=True):
        super().__init__(obj, config, SapphireConfig, detach)
    
    def init_state(self, variable_values: TensorDict)->SapphireState:
        return super().init_state(variable_values)

    def step(
            self, variable_values: TensorDict, state: SapphireState
    )-> tuple[TensorDict, SapphireState]:
        return super().step(variable_values, state)
    
    def solve(
            self, 
            variable_values: TensorDict | None = None, 
            stopping_criteria: GradSolverStoppingCriteria = GradSolverStoppingCriteria()
    ):
        return super().solve(variable_values, stopping_criteria)

def _split_objective(obj: Expression, config: GradSolverConfig):
    if isinstance(config, ProxGradConfig):
        return ProxGradSplit(obj)
    else:
        return SapphireSplit(obj)