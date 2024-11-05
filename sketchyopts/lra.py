import jax.numpy as jnp
import jax.scipy as jsp
from sketchyopts.base import KeyArray, Array
from sketchyopts.linop import ExtendedLinearOperator
from sketchyopts import sketches
from jax.lax import cond
from typing import Any, Tuple, Union 

def _shifted_cholesky(target, shift):
    """Cholesky factorization on a matrix with shifted eigenvalues (so that it is
    positive definite if ``shift`` is positive)."""
    eigs, eigvectors = jsp.linalg.eigh(target)
    new_shift = shift + jnp.abs(jnp.min(eigs))
    L = jsp.linalg.cholesky(
        eigvectors @ jnp.diag(eigs + new_shift) @ eigvectors.T, lower=False
    )
    return L, new_shift

def rand_nystrom_approx(A: Union[jnp.ndarray, ExtendedLinearOperator], 
                        sketch_size: int, 
                        key: KeyArray,
                        sketch_type: str,
                        is_array: bool,
                        *args: Any
                        ) -> Tuple[Array, Array]:
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
      - H\. Li, G. C. Linderman, A. Szlam, K. P. Stanton, Y. Kluger, and M. Tygert,
        `Algorithm 971: An implementation of a randomized algorithm for principal
        component analysis <https://dl.acm.org/doi/10.1145/3004053>`_, ACM Transactions
        on Mathematical Software (TOMS), 43: 1–14, 2017.
      - J\. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, `Fixed-rank approximation
        of a positive-semidefinite matrix from streaming data
        <https://papers.nips.cc/paper_files/paper/2017/hash/
        4558dbb6f6f8bb2e16d03b85bde76e2c-Abstract.html>`_, Advances in Neural
        Information Processing Systems, 30, 2017.

    Args:
      A: A two-dimensional array representing a positive-semidefinite matrix. This can
        either be an explicit JAX array or an implicit
        :class:`sketchyopts.util.LinearOperator` object.
      sketch_size: sketch_size used to construct the Nyström approximated matrix :math:`\hat{A}_{\mathrm{nys}}`. 
      The rank of resulting approximation satisfies rank<=sketch_size
      key: A PRNG key used as the random key.

    Returns:
      A two-element tuple containing

      - **U** – Orthonormal matrix :math:`U`.
      - **S** – Diagonal entries of :math:`\hat{\Lambda}`.

    """
    n=A.shape[1]
    # generate randomized sketch
    if sketch_type == 'gauss':
       S, _ = sketches.gauss_sketch_mat(key, (n,n), sketch_size, 'right')
    elif sketch_type == 'ortho':
       S,_ = sketches.ortho_sketch_mat(key, (n,n), sketch_size, 'right')  
    elif sketch_type == 'sjlt':
       S,_=sketches.sjlt_sketch_mat(key, (n,n), sketch_size,'right')
    else:
        raise ValueError(f"We do not support the sketch_type: {sketch_type}") 
    # shift Y for numerical stability
    if is_array:
      Y = A@S
    else:
      Y = A.matmul(S, *args) 
    
    shift = jnp.linalg.norm(Y, ord="fro")
    shift = shift - jnp.nextafter(shift, 0.0)
    Y_shifted = Y + shift * S
    # compute rank-l approximate Cholesky factor of Nyström approximation
    cholesky_target = S.T @ Y_shifted
    C = jsp.linalg.cholesky(cholesky_target, lower=False)

    # fail-safe step if Cholesky fails
    # Cholesky fails when the returned matrix has NaN values
    # This behavior is explained in post https://github.com/google/jax/issues/775

    C, shift = cond(
        jnp.any(jnp.isnan(C)),
        _shifted_cholesky,
        lambda x, y: (C, shift),
        cholesky_target,
        shift,
    )

    B = jsp.linalg.solve_triangular(C, Y_shifted.T, lower=False, trans=1)
    U, S, _ = jsp.linalg.svd(B.T, full_matrices=False)
    S = jnp.maximum(0.0, jnp.square(S) - shift)

    return U, S