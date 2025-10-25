"""Solver implementations and configurations."""

from rlaopt.solvers.configs_base import SolverConfig as SolverConfig
from rlaopt.solvers.configs_base import StoppingCriteria as StoppingCriteria
from rlaopt.solvers.pcg import PCG as PCG
from rlaopt.solvers.pcg import PCGConfig as PCGConfig
from rlaopt.solvers.pcg import PCGStoppingCriteria as PCGStoppingCriteria
from rlaopt.solvers.prox_grad import ProxGrad as ProxGrad
from rlaopt.solvers.prox_grad import ProxGradConfig as ProxGradConfig
from rlaopt.solvers.prox_grad import (
    ProxGradStoppingCriteria as ProxGradStoppingCriteria,
)
from rlaopt.solvers.solver_base import LinSysSolver as LinSysSolver
from rlaopt.solvers.solver_base import OptimSolver as OptimSolver
