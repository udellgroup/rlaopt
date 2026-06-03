"""Gradient solvers module."""

from dataclasses import dataclass
from time import perf_counter

from pydantic import Field

from rlaopt.expression import Expression
from rlaopt.ext_tensordict import TensorDict
from rlaopt.solvers.configs_base import StoppingCriteria
from rlaopt.solvers.solver_base import ConvergenceStatus, OptimResult, OptimSolver
from rlaopt.splitting import ProxGradSplit, SapphireSplit

from .gradient_solver_configs import GradSolverConfig, ProxGradConfig, SapphireConfig
from .gradient_solver_states import (
    GradSolverState,
    ProxGradState,
    SapphireState,
    get_solver_state,
)
from .step_builder import get_step_fn


class GradSolverStoppingCriteria(StoppingCriteria):
    """Stopping criteria for Gradient-based solvers.

    Convergence is declared when:

    ||G(x_k)|| <= eps_abs + eps_rel * ||x_k||.

    Attributes:
        max_iters: Maximum number of iterations.
        eps_abs: Absolute tolerance.
        eps_rel: Relative tolerance.
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
    """Base class for gradient-based solvers (ProxGrad, SAPPHIRE)."""

    def __init__(
        self,
        obj: Expression,
        config: GradSolverConfig,
        target_config: GradSolverConfig,
        detach=True,
    ):
        """Initialize the gradient solver.

        Args:
            obj: Optimization objective expression.
            config: Solver configuration.
            target_config: Expected configuration type for validation.
            detach: Whether to detach variable values from the autograd graph.
        """
        if not isinstance(obj, Expression):
            raise ValueError(
                f"{type(self).__name__} solver requires an Expression objective."
            )

        if not isinstance(config, target_config):
            raise ValueError(
                f"{type(self).__name__} solver requires an {target_config} config."
            )

        super().__init__(obj, config, detach)
        self._config = config
        self._detach = detach
        self._op_split = _split_objective(obj, config)
        self._step = get_step_fn(config, self._op_split)

    def init_state(self, variable_values: TensorDict) -> GradSolverState:
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
        stopping_criteria: GradSolverStoppingCriteria = GradSolverStoppingCriteria(),
    ) -> GradSolverResult:
        """Run the solver until convergence or maximum iterations."""
        ts = perf_counter()

        if variable_values is None:
            variable_values = self._op_split.variable_values

        state = self.init_state(variable_values)
        max_iters = stopping_criteria.max_iters
        eps_abs, eps_rel = stopping_criteria.eps_abs, stopping_criteria.eps_rel

        while (
            state.err > eps_abs + eps_rel * variable_values.flat_norm()
            and state.iter_ < max_iters
        ):
            variable_values, state = self.step(variable_values, state)

        if state.err <= eps_abs + eps_rel * variable_values.flat_norm():
            convergence_status = ConvergenceStatus.CONVERGED
        else:
            convergence_status = ConvergenceStatus.NOT_CONVERGED

        tf = perf_counter() - ts

        return GradSolverResult(
            variable_values,
            convergence_status,
            state.iter_,
            solver_time=tf,
            err=state.err,
        )

    def step(self, variable_values: TensorDict, state: GradSolverState):
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
    - Preconditioned proximal gradient (e.g. Nyström) with optional
      automatic stepsize
    - Combinations of acceleration and line search

    Combination constraints (enforced by ``ProxGradConfig``):
    - A non-identity preconditioner is incompatible with both line search
      and Nesterov acceleration.
    - Line search and automatic stepsize cannot both be enabled.
    """

    def __init__(self, obj: Expression, config: ProxGradConfig, detach=True):
        """Initialize the proximal gradient solver.

        Args:
            obj: The optimization objective (Expression).
            config: Configuration for the solver.
            detach: Whether to detach variable values from the autograd graph.
        """
        if not isinstance(config, ProxGradConfig):
            raise ValueError("ProxGrad solver requires a ProxGradConfig configuration.")
        super().__init__(obj, config, ProxGradConfig, detach)

    def init_state(self, variable_values: TensorDict) -> ProxGradState:
        """Initialize the proximal gradient solver state."""
        return super().init_state(variable_values)

    def step(
        self, variable_values: TensorDict, state: ProxGradState
    ) -> tuple[TensorDict, ProxGradState]:
        """Perform a single proximal gradient step."""
        return super().step(variable_values, state)


class Sapphire(BaseGradientSolver):
    r"""SAPPHIRE solver for empirical risk minimization.

    Solves problems of the form:

    .. math::
        minimize_{w} 1/n sum^{n}_{i=1}f_i(x_i^{T}\beta) + g(\beta) + r(\beta),

    here the first term is the ERM portion of the objective, g(w) is
    a smooth function and r(w) is a non-smooth proxable function.

    SAPPHIRE [6] is a meta algorithm that takes in a base stochastic gradient method:
    SGD [1], SVRG [2], or SAGA [3] and returns a scalable preconditioned variant.
    When the config uses an identity preconditioner,
    SAPPHIRE collapses down to the base method.
    When equipped with the Nyström preconditioner, SAPPHIRE recovers SketchySGD [4],
    PROMISE methods [5], and their proximal extensions when r is non-zero.

    Args:
        obj: Optimization objective of the form discussed above.
        config: Configuration for SAPPHIRE solver.

    Example:
        >>> import torch
        >>> from rlaopt.data import Dataset, DataLoader
        >>> from rlaopt.expression import Variable
        >>> from rlaopt.atoms import LinearRegression, L1Norm
        >>> from rlaopt.solvers import Sapphire, SapphireConfig

        >>> # Generate data
        >>> X, y = torch.randn(1024, 256), y = torch.randn(1024)

        >>> # Setup loader
        >>> dataset = Dataset(X,y)
        >>> loader = Dataloader(dataset,batch_size=256)

        >>> # Define objective.
        >>> beta = Variable((256,) name="beta")
        >>> model = LinearRegression(beta, loader)
        >>> obj = model + 0.001 * L1Norm(beta)

        >>> # Use SAPPHIRE with SVRG as the base with identity preconditioner
        >>> config = SapphireConfig(eta0=0.25,base_method='svrg')
        >>> opt = Sapphire(obj, config)

        >>> # Solve
        >>> result = opt.solve()

    **References**:
    .. [1] Bottou, L. (2010).
            Large-Scale Machine Learning with Stochastic Gradient Descent.
            International Conference on Computational Statistics

    .. [2] Johnson, R., Zhang T. (2013).
            Accelerating Stochastic Gradient Descent using Predictive Variance
            Reduction. Advances in Neural Information Processing Systems.

    .. [3] Defazio, A., Bach, F., Lacoste-Julien, S. (2014).
           SAGA: A Fast Incremental Gradient Method With
           Support for Non-Strongly Convex Composite Objectives.
           Advances in Neural Information Processing Systems.

    .. [4] Frangella, Z., Rathore, P., & Zhao, S., Udell, M. (2024).
            SketchySGD: Reliable Stochastic Optimization via Randomized Curvature
            Estimates. SIAM Journal on Mathematics of Data Science.

    .. [5] Frangella, Z., Rathore, P., & Zhao, S., Udell, M. (2024).
            PROMISE: Reliable Stochastic Optimization via Randomized Curvature
            Estimates. Journal of Machine Learning Research.

    .. [6] Sun, J., Frangella, Z., Udell, M. (2025).
           SAPPHIRE: Preconditioned Stochastic Variance Reduction for Faster
           Large-Scale Statistical Learning. Preprint.

    """

    def __init__(self, obj: Expression, config: SapphireConfig, detach=True):
        """Initialize the SAPPHIRE solver.

        Args:
            obj: Optimization objective of the form discussed in the class docstring.
            config: Configuration for SAPPHIRE solver.
            detach: Whether to detach variable values from the autograd graph.
        """
        super().__init__(obj, config, SapphireConfig, detach)

    def init_state(self, variable_values: TensorDict) -> SapphireState:
        """Initialize the SAPPHIRE solver state."""
        return super().init_state(variable_values)

    def step(
        self, variable_values: TensorDict, state: SapphireState
    ) -> tuple[TensorDict, SapphireState]:
        """Perform a single SAPPHIRE step."""
        return super().step(variable_values, state)

    def solve(
        self,
        variable_values: TensorDict | None = None,
        stopping_criteria: GradSolverStoppingCriteria = GradSolverStoppingCriteria(),
    ):
        """Run SAPPHIRE until convergence or max iterations."""
        return super().solve(variable_values, stopping_criteria)


def _split_objective(obj: Expression, config: GradSolverConfig):
    if isinstance(config, ProxGradConfig):
        return ProxGradSplit(obj)
    else:
        return SapphireSplit(obj)
