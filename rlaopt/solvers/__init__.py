"""Solver implementations and configurations."""

from rlaopt.solvers.admm import ADMM as ADMM
from rlaopt.solvers.admm import ADMMConfig as ADMMConfig
from rlaopt.solvers.admm import ADMMResult as ADMMResult
from rlaopt.solvers.admm import ADMMStoppingCriteria as ADMMStoppingCriteria
from rlaopt.solvers.configs_base import SolverConfig as SolverConfig
from rlaopt.solvers.configs_base import StoppingCriteria as StoppingCriteria
from rlaopt.solvers.pcg import PCG as PCG
from rlaopt.solvers.pcg import PCGConfig as PCGConfig
from rlaopt.solvers.pcg import PCGResult as PCGResult
from rlaopt.solvers.pcg import PCGStoppingCriteria as PCGStoppingCriteria
from rlaopt.solvers.prox_grad import ProxGrad as ProxGrad
from rlaopt.solvers.prox_grad import ProxGradConfig as ProxGradConfig
from rlaopt.solvers.prox_grad import ProxGradResult as ProxGradResult
from rlaopt.solvers.prox_grad import (
    ProxGradStoppingCriteria as ProxGradStoppingCriteria,
)
from rlaopt.solvers.solver_base import ConvergenceStatus as ConvergenceStatus
from rlaopt.solvers.solver_base import LinSysResult as LinSysResult
from rlaopt.solvers.solver_base import LinSysSolver as LinSysSolver
from rlaopt.solvers.solver_base import OptimResult as OptimResult
from rlaopt.solvers.solver_base import OptimSolver as OptimSolver
