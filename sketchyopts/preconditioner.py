import jax.random
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from sketchyopts.errors import InputDimError, MatrixNotSquareError

from jax.typing import ArrayLike
from jax import Array

KeyArrayLike = ArrayLike


def rand_nystrom_approx(A: ArrayLike, l: int, key: KeyArrayLike) -> tuple[Array, Array]:
    r"""Randomized Nyström approximation of a positive-semidefinite matrix.

    The function computes rank-:math:`l` randomized Nyström approximation of a given positive-semidefinite matrix :math:`A \in \R^{n \times n}`.

    It returns the truncated eigen-decomposition :math:`\hat{A}_{\mathrm{nys}} = U \hat{\Lambda} U^{T}` where :math:`U \in \R^{n \times l}` is orthonormal consisting of eigenvectors as columns and :math:`\hat{\Lambda} \in \R^{l \times l}` is diagonal with eigenvalues on its diagonal. Note that output :code:`U` represents :math:`U` whereas :code:`S` contains only the eigenvalues (not the full diagonal matrix :math:`\hat{\Lambda}`).

    References:
      - H\. Li, G. C. Linderman, A. Szlam, K. P. Stanton, Y. Kluger, and M. Tygert, `Algorithm 971: An implementation of a randomized algorithm for principal component analysis <https://dl.acm.org/doi/10.1145/3004053>`_, ACM Transactions on Mathematical Software (TOMS), 43 (2017), pp. 1–14.
      - J\. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, `Fixed-rank approximation of a positive-semidefinite matrix from streaming data <https://dl.acm.org/doi/10.5555/3294771.3294888>`_, in NIPS, vol. 30, 2017, pp. 1225–1234.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix.
      l: Rank of the Nyström approximated matrix :math:`\hat{A}_{\mathrm{nys}}` (which coincides with sketch size).
      key: A PRNG key used as the random key.

    Returns:
      A two-element tuple containing

      - **U** – orthonormal matrix :math:`U`.
      - **S** – diagonal entries of :math:`\hat{\Lambda}`.

    """
    # dimension check
    if jnp.ndim(A) != 2:
        raise InputDimError("A", jnp.ndim(A), 2)
    else:
        if jnp.shape(A)[0] != jnp.shape(A)[1]:
            raise MatrixNotSquareError("A", jnp.shape(A))
    # generate randomized sketch
    Omega = jax.random.normal(key, (jnp.shape(A)[0], l))
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

    return (U, S)
