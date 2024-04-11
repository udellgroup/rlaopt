import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.flatten_util import ravel_pytree
import numpy as np
from optax._src.base import GradientTransformationExtraArgs, EmptyState
from optax._src import numerics

from sketchyopts.util import (
    LinearOperator,
    HessianLinearOperator,
    GradientTransformationExtraArgsRefState,
)
from sketchyopts.errors import InputDimError, MatrixNotSquareError

from typing import NamedTuple, Union, Callable, Optional, ParamSpec, Concatenate
from jax.typing import ArrayLike
from jax import Array

KeyArray = Array
KeyArrayLike = ArrayLike
P = ParamSpec("P")


def rand_nystrom_approx(
    A: Union[ArrayLike, LinearOperator], l: int, key: KeyArrayLike
) -> tuple[Array, Array]:
    r"""Randomized Nyström approximation of a positive-semidefinite matrix.

    The function computes rank-:math:`l` randomized Nyström approximation of a given
    positive-semidefinite matrix :math:`A \in \R^{n \times n}`.

    It returns the truncated eigen-decomposition
    :math:`\hat{A}_{\mathrm{nys}} = U \hat{\Lambda} U^{T}` where
    :math:`U \in \R^{n \times l}` is orthonormal consisting of eigenvectors as columns
    and :math:`\hat{\Lambda} \in \R^{l \times l}` is diagonal with eigenvalues on its
    diagonal. Note that output :code:`U` represents :math:`U` whereas :code:`S` contains
    only the eigenvalues (not the full diagonal matrix :math:`\hat{\Lambda}`).

    References:
      - H\. Li, G. C. Linderman, A. Szlam, K. P. Stanton, Y. Kluger, and M. Tygert, `Algorithm 971: An implementation of a randomized algorithm for principal component analysis <https://dl.acm.org/doi/10.1145/3004053>`_, ACM Transactions on Mathematical Software (TOMS), 43 (2017), pp. 1–14.
      - J\. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, `Fixed-rank approximation of a positive-semidefinite matrix from streaming data <https://dl.acm.org/doi/10.5555/3294771.3294888>`_, in NIPS, vol. 30, 2017, pp. 1225–1234.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix. This can
        either be an explicit JAX array or an implicit
        :class:`sketchyopts.util.LinearOperator` object.
      l: Rank of the Nyström approximated matrix :math:`\hat{A}_{\mathrm{nys}}`
        (which coincides with sketch size).
      key: A PRNG key used as the random key.

    Returns:
      A two-element tuple containing

      - **U** – orthonormal matrix :math:`U`.
      - **S** – diagonal entries of :math:`\hat{\Lambda}`.

    """
    # dimension check
    dim = jnp.ndim(A)  # type: ignore
    shape = jnp.shape(A)  # type: ignore
    if dim != 2:
        raise InputDimError("A", dim, 2)
    else:
        if shape[0] != shape[1]:
            raise MatrixNotSquareError("A", shape)
    # generate randomized sketch
    n = shape[0]
    Omega = jax.random.normal(key, (n, l))
    Omega = jsp.linalg.qr(Omega, mode="economic")[0]
    Y = A @ Omega
    # shift Y for numerical stability
    shift = np.spacing(jnp.linalg.norm(Y, ord="fro"))
    Y_shifted = Y + shift * Omega
    # compute rank-l approximate Cholesky factor of Nyström approximation
    C = jsp.linalg.cholesky(Omega.T @ Y_shifted, lower=False)
    B = jsp.linalg.solve_triangular(C, Y_shifted.T, lower=False, trans=1)
    U, S, _ = jsp.linalg.svd(B.T, full_matrices=False)
    S = jnp.maximum(0.0, jnp.square(S) - shift)

    return U, S


class NystromPrecondState(NamedTuple):
    r"""State for the Nyström preconditioner."""

    step_count: ArrayLike
    r"""Step count (number of executed iterations)."""
    U: ArrayLike
    r"""Eigenvectors of the preconditioner."""
    S: ArrayLike
    r"""Eigenvalues of the preconditioner."""
    learning_rate: Optional[ArrayLike]
    r"""Learning rate picked for the preconditioner."""
    key: KeyArrayLike
    r"""PRNG key to be used for updating the preconditioner."""


def update_nystrom_precond(
    rank: int,
    rho: float,
    update_freq: int,
    seed: int,
    f: Callable[Concatenate[Array, P], Array],
    *,
    adaptive_lr: bool = True,
) -> GradientTransformationExtraArgs:
    r"""Updates Nyström preconditioner.

    The function updates the preconditioner using randomized low-rank Nyström
    approximations to the subsampled Hessian and determines appropriate learning rate
    using randomized powering (if ``adaptive_lr`` is set to ``True``).

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates <https://arxiv.org/abs/2211.08597>`_.

    Args:
      rank: Rank of the preconditioner.
      rho: Regularization parameter (with non-negative value).
      update_freq: A non-negative integer that specifies the update frequency of the
        preconditioner. When set to ``0`` or :math:`\infty` (*e.g.* ``jax.numpy.inf`` or
        ``numpy.inf``), the optimizer uses constant preconditioner that is constructed at
        the beginning of the optimization process.
      seed: An integer used as a seed to generate random numbers.
      f: A scalar-valued loss/objective function to be optimized. The function needs to
        have the optimization variable as its first argument.
      adaptive_lr: Whether or not to enable automated learning rate selection (default
         ``True``). When set to ``True``, the algorithm adaptively chooses a learning rate
         whenever the preconditioner is updated using **Algorithm 2.1** of the referenced paper.

    Returns:
      A `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ object.
    """

    def init_fn(params):
        return NystromPrecondState(
            step_count=jnp.zeros([], jnp.int32),
            U=jnp.zeros([jnp.size(params), rank]),
            S=jnp.zeros(
                [
                    rank,
                ]
            ),
            learning_rate=None,
            key=jax.random.PRNGKey(seed),
        )

    # matrix-vector product for the inverse square-root of preconditioner
    def _sqrt_matvec(U, S, rho, v):
        UTv = U.T @ v
        return U @ (UTv / jnp.sqrt(S + rho)) + (1 / rho**0.5) * (v - U @ UTv)

    def _get_learning_rate(H_S, U, S, rho, key, tol: float = 1e-5, maxiter: int = 10):
        n = jnp.shape(H_S)[0]

        # stopping criterion
        def cond_fun(value):
            _, _, norm_r, k = value
            return (norm_r >= tol) & (k < maxiter)

        # power iteration
        def body_fun(value):
            y, _, _, k = value
            y_ = _sqrt_matvec(U, S, rho, y)
            y_ = H_S @ y_
            y_ = _sqrt_matvec(U, S, rho, y_)
            labda = jnp.dot(y, y_)
            norm_r = jnp.linalg.norm(y_ - labda * y)
            y = y_ / jnp.linalg.norm(y_)
            return y, labda, norm_r, k + 1

        # initialization
        y = jax.random.normal(key, (n,))
        y = y / jnp.linalg.norm(y)
        initial_value = (y, 0, tol, 0)

        _, labda, _, _ = jax.lax.while_loop(cond_fun, body_fun, initial_value)

        return 1.0 / labda

    def update_fn(updates, state, params, **f_extra_args):
        # increment counter
        count_inc = numerics.safe_int32_increment(state.step_count)
        # update the preconditioner at the beginning and at the specified frequency
        if (state.step_count == 0) or (
            (update_freq > 0) & (state.step_count % update_freq == 0)
        ):
            key, subkey1, subkey2 = jax.random.split(state.key, num=3)
            H_S = HessianLinearOperator(f, params, **f_extra_args)
            U, S = rand_nystrom_approx(H_S, rank, subkey1)
            learning_rate = None
            if adaptive_lr:
                learning_rate = _get_learning_rate(H_S, U, S, rho, subkey2)
            return updates, NystromPrecondState(
                step_count=count_inc, U=U, S=S, learning_rate=learning_rate, key=key
            )
        # otherwise keep the preconditioner state as is (except for the step counter)
        else:
            return updates, NystromPrecondState(
                step_count=count_inc,
                U=state.U,
                S=state.S,
                learning_rate=state.learning_rate,
                key=state.key,
            )

    return GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore


def scale_by_nystrom_precond(
    rho: float,
    *,
    U: Optional[ArrayLike] = None,
    S: Optional[ArrayLike] = None,
    ref_state: Optional[str] = None,
) -> GradientTransformationExtraArgs:
    r"""Rescales updates using Nyström preconditioner.

    The function scales the gradient update using Nyström preconditioner. Specifically,
    it applies the
    `matrix inversion lemma <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_
    and computes the following (**Equation (2.3)** of the referenced paper):

    .. math::
      v = U(S + \rho I)^{-1} U^{T} g + \frac{1}{\rho} (g - U U^{T} g)

    where :math:`g` is the gradient and :math:`v` is the obtained update direction (from
    rescaling).

    If the eigendecomposition of the preconditioner is not directly provided (*i.e.*
    ``U`` and ``S`` are set to ``None``), the function then uses the preconditioner stored
    in the referenced state to scale the updates.

    References:
      - Z\. Frangella, P. Rathore, S. Zhao, and M. Udell, `SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates <https://arxiv.org/abs/2211.08597>`_.

    Args:
      rho: Regularization parameter (with non-negative value).
      U: Eigenvectors of the preconditioner.
      S: Eigenvalues of the preconditioner.
      ref_state: The name of the state in a named chain that provides the preconditioner.

    Returns:
      A `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ object.
    """

    if ((U is None) or (S is None)) and (ref_state is None):
        raise ValueError("U or S and ref_state cannot both be None")

    def init_fn(params):
        del params
        return EmptyState()

    def _apply_precond(U, S, updates):
        raveled, unravel_fn = ravel_pytree(updates)
        UTg = U.T @ raveled
        return unravel_fn(U @ (UTg / (S + rho)) + raveled / rho - U @ UTg / rho)

    if (U is not None) and (S is not None):

        def update_fn(updates, state, params=None):
            del params
            return _apply_precond(U, S, updates), state

        return GradientTransformationExtraArgs(init_fn, update_fn)
    else:

        def ref_update_fn(updates, state, params=None, chain_state=None):
            del params
            # validate referenced preconditioner
            if (chain_state is None) or (ref_state not in chain_state.keys()):
                raise ValueError("ref_state does not exist")
            if ("U" not in dir(chain_state[ref_state])) or (
                "S" not in dir(chain_state[ref_state])
            ):
                raise ValueError("U or S is not an attribute of the referenced state")
            if (not isinstance(chain_state[ref_state].U, Array)) or (
                not isinstance(chain_state[ref_state].S, Array)
            ):
                raise ValueError("U or S has incompatible type")
            return (
                _apply_precond(
                    chain_state[ref_state].U, chain_state[ref_state].S, updates
                ),
                state,
            )

        return GradientTransformationExtraArgsRefState(init_fn, ref_update_fn)
