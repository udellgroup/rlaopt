import jax
import jax.numpy as jnp
import lineax as lx

from sketchyopts.sketching import SKETCH_TYPE_OPTIONS
from sketchyopts.util import is_array

from .solver.lsqr import abstract_lsqr


def sketch_and_solve(
    operator,
    right_hand_side,
    solver,
    sketch_size,
    *,
    seed=0,
    solver_args=None,
    solver_state=None,
    sketch_type="gaussian",
    sketch_args=None,
):
    r"""Solve a linear system using the sketch-and-solve approach.

    The function approximates the solution to the linear system :math:`Ax = b` by
    sketching the operator and right-hand side, and then solving the sketched linear
    system.

    Args:
      operator: Linear operator of the system. This is the :math:`A` matrix in the
        system :math:`Ax = b`.
      right_hand_side: Right-hand side of the system. This is the :math:`b` vector in
        the system :math:`Ax = b`.
      solver: Solver to use for solving the linear system.
      sketch_size: Size of the sketch.
      seed: Random seed for the sketching matrix (default ``0``).
      solver_args: Optional arguments for the solver. Keyword-only argument.
      solver_state: Optional state for the solver. This intends to re-use intermediate
        computation between multiple linear solves with the same operator. Keyword-only
        argument.
      sketch_type: Type of sketching matrix to use (default ``gaussian``). Options are
        ``"gaussian"`` (Gaussian embedding), ``"srtt"`` (subsampled randomized
        trigonometric transforms), and ``"sparse-sign"`` (sparse sign embedding).
      sketch_args: Optional arguments for the sketch method. Keyword-only argument.

    Returns:
      Solution of the linear system that contains the following attributes:
        - ``value``: Solution to the linear system.
        - ``result``: An integer representing whether the solve was successful or not.
        - ``stats``: Statistics of the solver after solving the linear system.
        - ``state``: State of the solver after solving the linear system.
    """
    # check sketch type and sketch method
    if sketch_type not in SKETCH_TYPE_OPTIONS:
        raise ValueError(f"Unknown sketch type: {sketch_type}")

    # initialize sketched operator
    key = jax.random.PRNGKey(seed)
    if is_array(operator):
        operator = lx.MatrixLinearOperator(operator)
    sketched_op = SKETCH_TYPE_OPTIONS["sketch_type"](
        operator, sketch_size, key, **sketch_args
    )

    return lx._solve.linear_solve(
        sketched_op,
        sketched_op.apply_sketch(right_hand_side),
        solver,
        options=solver_args,
        state=solver_state,
    )


def sketch_and_precondition(
    operator,
    right_hand_side,
    sketch_size,
    *,
    seed=0,
    sketch_type="gaussian",
    sketch_args=None,
    tol=1e-6,
    maxiter=None,
):
    r"""Solve a linear system using the sketch-and-precondition approach.

    The function approximates the solution to the linear system :math:`Ax = b` by
    sketching the operator and use the preconditioned iterative solver to solve the
    system.

    References:
      - V\. Rokhlin and M. Tygert, `A fast randomized algorithm for overdetermined linear
        least-squares regression <https://doi.org/10.1073/pnas.0804869105>`_. The
        Proceedings of the National Academy of Sciences, 105(36), 2008.

    Args:
      operator: Linear operator of the system. This is the :math:`A` matrix in the
        system :math:`Ax = b`.
      right_hand_side: Right-hand side of the system. This is the :math:`b` vector in
        the system :math:`Ax = b`.
      sketch_size: Size of the sketch.
      seed: Random seed for the sketching matrix (default ``0``).
      sketch_type: Type of sketching matrix to use (default ``gaussian``). Options are
        ``"gaussian"`` (Gaussian embedding), ``"srtt"`` (subsampled randomized
        trigonometric transforms), and ``"sparse-sign"`` (sparse sign embedding).
      sketch_args: Optional arguments for the sketch method. Keyword-only argument.
      tol: Solution tolerance (default `1e-6`).
      maxiter: Maximum number of iterations for the solver (default `None`). If `None`,
        the solver will use :math:`2` times the sketch size as its value (same behavior
        as SciPy LSQR implementation).

    Returns:
      Approximate solution to the linear least-squares problem and the number of
        iterations.

    """
    # check sketch type and sketch method
    if sketch_type not in SKETCH_TYPE_OPTIONS:
        raise ValueError(f"Unknown sketch type: {sketch_type}")

    # set maximum number of iterations
    if maxiter is None:
        maxiter = 2 * sketch_size

    # apply sketch
    key = jax.random.PRNGKey(seed)
    if is_array(operator):
        operator = lx.MatrixLinearOperator(operator)
    sketched_op = SKETCH_TYPE_OPTIONS["sketch_type"](
        operator, sketch_size, key, **sketch_args
    )
    sketched_rhs = sketched_op.apply_sketch(right_hand_side)

    # compute initial guess
    Q, R = jnp.linalg.qr(sketched_op.as_matrix(), mode="reduced")
    x0 = jnp.matmul(Q.T, sketched_rhs)

    # solve the problem with preconditioned solver
    def M(x):
        return jax.scipy.linalg.solve_triangular(R, x, lower=False)

    def M_transpose(x):
        return jax.scipy.linalg.solve_triangular(R.T, x, lower=True)

    x_final, _, _, num_iter = abstract_lsqr(
        operator, right_hand_side, x0, tol, maxiter, M, M_transpose
    )

    return x_final, num_iter
