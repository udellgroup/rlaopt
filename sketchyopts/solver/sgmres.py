import jax
import jax.numpy as jnp
import lineax as lx

from sketchyopts.sketching import SKETCH_TYPE_OPTIONS
from sketchyopts.util import is_array


def sgmres(
    operator,
    right_hand_side,
    d,
    num_ortho_vectors,
    tol,
    maxiter,
    *,
    seed=0,
    sketch_type="gaussian",
    sketch_args=None,
    x0=None,
):
    r"""Solve a linear system using the sketched GMRES (sGMRES) algorithm.

    References:
      - Y\. Nakatsukasa and J. A. Tropp, `Fast and Accurate Randomized Algorithms for
        Linear Systems and Eigenvalue Problems <https://doi.org/10.1137/23M1565413>`_.
        SIAM Journal on Matrix Analysis and Applications, vol. 45, iss. 2, 2024, pp.
        1183-1214.

    Args:
      operator: Linear operator of the system.
      right_hand_side: Right-hand side of the system.
      d: Dimension of the Krylov subspace.
      num_ortho_vectors: Number of Arnoldi process steps to run.
      tol: Solution tolerance.
      maxiter: Maximum number of iterations.
      seed: Random seed for the sketching matrix (default ``0``).
      sketch_type: Type of sketching matrix to use (default ``gaussian``).
      sketch_args: Optional arguments for the sketch method. Keyword-only argument.
      x0: Initial guess for the solution (default ``None``).

    Returns:
      Approximate solution to the linear system.
    """
    raise NotImplementedError("sGMRES is not implemented yet.")
