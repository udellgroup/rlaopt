from typing import Any, NamedTuple

from jax import Array
from jax.typing import ArrayLike

KeyArray = Array
KeyArrayLike = ArrayLike


class LinearSolveState(NamedTuple):
    r"""The linear solve state.

    Args:
      iter_num: Number of iterations the solver has performed.
      maxiter: Maximum number of iterations.
      residual: residual of the solution.
    """

    iter_num: int
    maxiter: int
    residual: Array


class SolverState(NamedTuple):
    r"""Class for encapsulating parameters and solver state.

    Args:
      params: Parameters the solver seeks to optimize.
      state: State of the solver.
    """

    params: Any
    state: Any
