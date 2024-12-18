from abc import ABC, abstractmethod
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array
from lineax import AbstractLinearOperator, IdentityLinearOperator, MatrixLinearOperator

from sketchyopts.util import default_floating_dtype, ravel_tree, sample_indices


class Sketching(ABC):
    r"""Abstract class for sketching matrix.

    The class defines the interface for a sketching matrix. The class represents a
    sketching matrix that can be applied to a linear operator or a vector. For a
    left-sketching matrix, the size is :math:`(\text{sketch\_size}, \text{dim})`;
    for a right-sketching matrix, the size is :math:`(\text{dim}, \text{sketch\_size})`.

    Args:
      sketch_size: Size of the sketch.
      dim: Dimension of the input or output space for a left- or right-sketching matrix
        respectively.
      key: Seed for randomization.
      mode: Sketching mode (default ``left``). The mode can be either ``left`` for
        left-sketching or ``right`` for right-sketching.
    """

    def __init__(self, sketch_size: int, dim: int, seed: int, mode: str = "left"):
        r"""Initialize the sketching matrix."""
        # validate arguments
        if sketch_size < 1:
            raise ValueError("Sketch size cannot be less than 1.")
        if dim < 1:
            raise ValueError("Dimension cannot be less than 1.")
        if mode not in ["left", "right"]:
            raise ValueError("Invalid sketching mode.")
        # set attributes
        self.sketch_size = sketch_size
        self.dim = dim
        self.key = jax.random.PRNGKey(seed)
        self.mode = mode

    @abstractmethod
    def _generate_sketching(self) -> None:
        r"""Internal function for generate the sketching matrix."""
        pass

    @abstractmethod
    def as_matrix(self) -> Array:
        r"""Materialize the sketching matrix."""
        pass

    @abstractmethod
    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        r"""Apply the sketching matrix to a linear operator."""
        pass

    @abstractmethod
    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        r"""Apply the sketching matrix to a vector."""
        pass


class GaussianEmbedding(Sketching):
    r"""Gaussian random matrix.

    This sketching is defined by a Gaussian random matrix where each entry is
    independently drawn from a normal distribution with zero mean and variance
    :math:`1 / \text{sketch\_size}`.
    """

    def __init__(self, sketch_size: int, dim: int, seed: int, mode: str = "left"):
        super().__init__(sketch_size, dim, seed, mode)
        self._generate_sketching()

    def _generate_sketching(self) -> None:
        # split key
        self.key, sub_key = jax.random.split(self.key)
        if self.mode == "left":
            self.matrix = (1.0 / jnp.sqrt(self.sketch_size)) * jax.random.normal(
                sub_key, (self.sketch_size, self.dim)
            )
        else:
            self.matrix = (1.0 / jnp.sqrt(self.dim)) * jax.random.normal(
                sub_key, (self.dim, self.sketch_size)
            )

    def as_matrix(self) -> Array:
        return self.matrix

    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        # check dimension
        if (self.mode == "left") and (matrix.out_size() != self.dim):
            raise ValueError(
                "Output dimension of the linear operator does not match the input dimension of the sketching matrix."
            )
        if (self.mode == "right") and (matrix.in_size() != self.dim):
            raise ValueError(
                "Input dimension of the linear operator does not match the output dimension of the sketching matrix."
            )
        # generate new sketching matrix if needed
        if new_sketch or not hasattr(self, "matrix"):
            self._generate_sketching()
        # return sketched matrix
        if self.mode == "left":
            return MatrixLinearOperator(
                self.as_matrix() @ matrix.as_matrix()
            ) @ IdentityLinearOperator(
                matrix.in_structure(),
                jax.ShapeDtypeStruct(
                    shape=(matrix.in_size(),), dtype=default_floating_dtype()
                ),
            )
        else:
            return IdentityLinearOperator(
                matrix.out_structure(),
                jax.ShapeDtypeStruct(
                    shape=(matrix.out_size(),), dtype=default_floating_dtype()
                ),
            ) @ MatrixLinearOperator(matrix.as_matrix() @ self.as_matrix())

    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        # check mode
        if self.mode == "right":
            raise ValueError(
                "Sketch a vector with a right-sketching matrix is not allowed."
            )
        # ravel input vector
        unraveled, _ = ravel_tree(vector)
        if new_sketch or not hasattr(self, "matrix"):
            self._generate_sketching()
        return jnp.matmul(
            self.as_matrix(), unraveled, precision=jax.lax.Precision.HIGHEST
        )


class OrthonormalEmbedding(Sketching):
    r"""Orthonormal random matrix.

    This sketching is defined by an orthonormal random matrix where each column is an
    orthonormal basis.
    """

    def __init__(self, sketch_size: int, dim: int, seed: int, mode: str = "left"):
        super().__init__(sketch_size, dim, seed, mode)
        self._generate_sketching()

    def _generate_sketching(self) -> None:
        # split key
        self.key, sub_key = jax.random.split(self.key)
        if self.mode == "left":
            self.matrix = jsp.linalg.qr(
                (1.0 / jnp.sqrt(self.sketch_size))
                * jax.random.normal(sub_key, (self.dim, self.sketch_size)),
                mode="economic",
            )[0].T
        else:
            self.matrix = jsp.linalg.qr(
                (1.0 / jnp.sqrt(self.dim))
                * jax.random.normal(sub_key, (self.dim, self.sketch_size)),
                mode="economic",
            )[0]

    def as_matrix(self) -> Array:
        return self.matrix

    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        # check dimension
        if (self.mode == "left") and (matrix.out_size() != self.dim):
            raise ValueError(
                "Output dimension of the linear operator does not match the input dimension of the sketching matrix."
            )
        if (self.mode == "right") and (matrix.in_size() != self.dim):
            raise ValueError(
                "Input dimension of the linear operator does not match the output dimension of the sketching matrix."
            )
        # generate new sketching matrix if needed
        if new_sketch or not hasattr(self, "matrix"):
            self._generate_sketching()
        # return sketched matrix
        if self.mode == "left":
            return MatrixLinearOperator(
                self.as_matrix() @ matrix.as_matrix()
            ) @ IdentityLinearOperator(
                matrix.in_structure(),
                jax.ShapeDtypeStruct(
                    shape=(matrix.in_size(),), dtype=default_floating_dtype()
                ),
            )
        else:
            return IdentityLinearOperator(
                matrix.out_structure(),
                jax.ShapeDtypeStruct(
                    shape=(matrix.out_size(),), dtype=default_floating_dtype()
                ),
            ) @ MatrixLinearOperator(matrix.as_matrix() @ self.as_matrix())

    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        # check mode
        if self.mode == "right":
            raise ValueError(
                "Sketch a vector with a right-sketching matrix is not allowed."
            )
        # ravel input vector
        unraveled, _ = ravel_tree(vector)
        # check dimension
        if len(unraveled) != self.dim:
            raise ValueError(
                "Vector length does not match the input dimension of the sketching matrix."
            )
        if new_sketch or not hasattr(self, "matrix"):
            self._generate_sketching()
        return jnp.matmul(
            self.as_matrix(), unraveled, precision=jax.lax.Precision.HIGHEST
        )


class SJLT(Sketching):
    r"""Sparse Johnson–Lindenstrauss transform (SJLT) matrix.

    This sketching is defined by a sparse Johnson–Lindenstrauss transform (SJLT) matrix
    where each row has exactly :math:`\zeta` non-zero entries. Each non-zero entry is
    independently drawn from a Rademacher distribution (uniform over :math:`\{\pm 1\}`),
    and its location in the row is randomly chosen from
    :math:`\{1, \cdots, \text{dim}\}`.

    The sketching has an additional optional argument ``sparsity`` that specifies the
    value of :math:`\zeta`. By default (``None``), the sparsity parameter :math:`\zeta`
    takes the recommended practical value of
    :math:`\operatorname{min}\{8, \text{sketch\_size}\}` [#f6]_.

    .. rubric:: References

    .. [#f6]  J\. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, `Streaming Low-Rank
      Matrix Approximation with an Application to Scientific Simulation <https://epubs.
      siam.org/doi/10.1137/18M1201068>`_, SIAM Journal on Scientific Computing, 41(4),
      2019.
    """

    def __init__(
        self,
        sketch_size: int,
        dim: int,
        seed: int,
        mode: str = "left",
        sparsity: Optional[int] = None,
    ):
        super().__init__(sketch_size, dim, seed, mode)
        # set sparsity
        if sparsity is not None:
            if sparsity < 1:
                raise ValueError("Sparsity parameter cannot be less than 1.")
            if sparsity > self.sketch_size:
                raise ValueError("Sparsity parameter cannot exceed the sketch size.")
            self.sparsity = sparsity
        else:
            self.sparsity = min(8, self.sketch_size)
        self._generate_sketching()

    def _generate_sketching(self) -> None:
        # split keys
        self.key, sub_key_1, sub_key_2 = jax.random.split(self.key, num=3)
        # generate random coordinates for non-zero entries
        key_array = jax.random.split(sub_key_1, num=self.sketch_size)
        self.coord_matrix = jax.vmap(sample_indices, in_axes=(None, None, 0))(
            self.dim, self.sparsity, key_array
        )
        # generate random signs
        self.sign_matrix = jax.random.rademacher(
            sub_key_2, (self.sketch_size, self.sparsity)
        ) * jnp.sqrt(1.0 / self.sparsity)

    def as_matrix(self) -> Array:
        # initialize a zero matrix
        S = jnp.zeros((self.sketch_size, self.dim))
        # fill the matrix with non-zero entries
        S = S.at[jnp.arange(self.sketch_size)[:, None], self.coord_matrix].set(
            self.sign_matrix
        )
        if self.mode == "left":
            return S
        else:
            return S.T

    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        # check dimension
        if (self.mode == "left") and (matrix.out_size() != self.dim):
            raise ValueError(
                "Output dimension of the linear operator does not match the input dimension of the sketching matrix."
            )
        if (self.mode == "right") and (matrix.in_size() != self.dim):
            raise ValueError(
                "Input dimension of the linear operator does not match the output dimension of the sketching matrix."
            )
        # generate new sketching matrix if needed
        if new_sketch:
            self._generate_sketching()
        # return sketched matrix
        if self.mode == "left":
            return MatrixLinearOperator(
                self.as_matrix() @ matrix.as_matrix()
            ) @ IdentityLinearOperator(
                matrix.in_structure(),
                jax.ShapeDtypeStruct(
                    shape=(matrix.in_size(),), dtype=default_floating_dtype()
                ),
            )
        else:
            return IdentityLinearOperator(
                matrix.out_structure(),
                jax.ShapeDtypeStruct(
                    shape=(matrix.out_size(),), dtype=default_floating_dtype()
                ),
            ) @ MatrixLinearOperator(matrix.as_matrix() @ self.as_matrix())

    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        # check mode
        if self.mode == "right":
            raise ValueError(
                "Sketch a vector with a right-sketching matrix is not allowed."
            )
        # ravel input vector
        unraveled, _ = ravel_tree(vector)
        # check dimension
        if len(unraveled) != self.dim:
            raise ValueError(
                "Vector length does not match the input dimension of the sketching matrix."
            )
        if new_sketch:
            self._generate_sketching()
        return jnp.sum(self.sign_matrix * unraveled[self.coord_matrix], axis=1)


class SRTT(Sketching):
    r"""Subsampled randomized trigonometric transform (SRTT) sketching matrix.

    This sketching is defined by a random restriction :math:`R`, a discrete cosine
    transform :math:`F`, a random sign flip :math:`E`, and a random permutation
    :math:`\Gamma`:

    .. math::
        S = \sqrt{\frac{1}{\text{sketch\_size}}} R F E \Gamma

    Here random random restriction selects uniformly :math:`\text{sketch\_size}` random
    entries from its input; sign flip is a diagonal matrix such that its diagonal
    entries are i.i.d Rademacher random variables (*i.e.* random variable with equal
    probability mass assigned to :math:`\{\pm 1\}`); random permutation is a square
    matrix drawn uniformly at random from the set of :math:`\text{dim}`-permutation
    matrices.
    """

    def __init__(self, sketch_size: int, dim: int, seed: int, mode: str = "left"):
        if mode == "right":
            raise ValueError("Right-sketching with SRTT is not supported.")
        super().__init__(sketch_size, dim, seed, mode)
        self._generate_sketching()

    def _generate_sketching(self) -> None:
        # split keys
        self.key, sub_key_1, sub_key_2, sub_key_3 = jax.random.split(self.key, num=4)
        # generate sketching components
        self.permutation = jax.random.permutation(sub_key_1, self.dim)
        self.sign_flip = jax.random.rademacher(sub_key_2, (self.dim,))
        self.restriction = jax.random.choice(
            sub_key_3, self.dim, (self.sketch_size,), replace=True
        )

    def as_matrix(self) -> Array:
        raise ValueError("Materializing the SRTT sketching matrix is not supported.")

    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        # check dimension
        if (self.mode == "left") and (matrix.out_size() != self.dim):
            raise ValueError(
                "Output dimension of the linear operator does not match the input dimension of the sketching matrix."
            )
        if (self.mode == "right") and (matrix.in_size() != self.dim):
            raise ValueError(
                "Input dimension of the linear operator does not match the output dimension of the sketching matrix."
            )
        # generate new sketching matrix if needed
        if new_sketch:
            self._generate_sketching()
        # return sketched matrix
        if self.mode == "left":
            SA = jnp.take(
                matrix.as_matrix(), self.permutation, axis=0, unique_indices=True
            )
            SA = jnp.multiply(self.sign_flip[:, None], SA)
            SA = jsp.fft.dct(SA, axis=0)
            SA = jnp.take(SA, self.restriction, axis=0, unique_indices=False)
            return MatrixLinearOperator(
                jnp.sqrt(1 / self.sketch_size) * SA
            ) @ IdentityLinearOperator(
                jax.ShapeDtypeStruct(
                    shape=(matrix.in_size(),), dtype=default_floating_dtype()
                ),
                matrix.in_structure(),
            )
        else:
            raise ValueError("Right-sketching with SRTT is not supported.")

    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        # check mode
        if self.mode == "right":
            raise ValueError(
                "Sketch a vector with a right-sketching matrix is not allowed."
            )
        # ravel input vector
        unraveled, _ = ravel_tree(vector)
        # check dimension
        if len(unraveled) != self.dim:
            raise ValueError(
                "Vector length does not match the input dimension of the sketching matrix."
            )
        if new_sketch:
            self._generate_sketching()
        # compute sketched vector
        v = unraveled[self.permutation]
        v = self.sign_flip * v
        v = jsp.fft.dct(v)
        v = jnp.take(v, self.restriction, unique_indices=False)
        return jnp.sqrt(1 / self.sketch_size) * v


def hadamard_transform(n: int, M: Array) -> Array:
    r"""Hadamard transform of a matrix.

    The function computes the unscaled (without the :math:`n^{-1/2}` factor) Hadamard
    transform :math:`H_n M` using recursive construction (*i.e.* fast Walsh–Hadamard
    transform or Sylvester's construction).

    .. note::
        The function is intended for internal use, as it lacks validation for the value
        of ``n`` and the size of the matrix ``M``. The sketching matrix classes
        :class:`SRHT` and :class:`LESS`, which use this function, additionally handle
        validation, padding, and scaling.

    Args:
      n: Order of the Hadamard matrix. The value needs to be a power of 2.
      M: 2-dimensional array the transformation applies to.

    Returns:
      Transformed 2-dimensional array.
    """
    if n == 1:
        return M
    return jnp.vstack(
        [
            hadamard_transform(n // 2, M[: n // 2, ::] + M[n // 2 :, ::]),
            hadamard_transform(n // 2, M[: n // 2, ::] - M[n // 2 :, ::]),
        ]
    )


class SRHT(Sketching):
    r"""Subsampled randomized Hadamard transform (SRHT) sketching matrix.

    This sketching is defined by a random restriction :math:`R`, a Hadamard transform
    :math:`H`, a random sign flip :math:`E`:

    .. math::
        S = \sqrt{\frac{1}{\text{sketch\_size}}} R H E

    Here random random restriction selects uniformly :math:`\text{sketch\_size}` random
    entries from its input; sign flip is a diagonal matrix such that its diagonal
    entries are i.i.d Rademacher random variables (*i.e.* random variable with equal
    probability mass assigned to :math:`\{\pm 1\}`).

    For a matrix with dimension :math:`\text{dim}` that is not a power of two (required
    for defining a Hadamard matrix), sketching is instead applied to a padded version of
    the matrix, where additional zero rows are appended.
    """

    def __init__(self, sketch_size: int, dim: int, seed: int, mode: str = "left"):
        if mode == "right":
            raise ValueError("Right-sketching with SRHT is not supported.")
        self.padded_dim = 2 ** np.ceil(np.log2(dim)).astype(np.int32)
        super().__init__(sketch_size, dim, seed, mode)
        self._generate_sketching()

    def _generate_sketching(self) -> None:
        # split keys
        self.key, sub_key_1, sub_key_2 = jax.random.split(self.key, num=3)
        # generate sketching components
        self.sign_flip = jax.random.rademacher(sub_key_1, (self.dim,)) * jnp.sqrt(
            self.padded_dim / self.sketch_size
        )
        self.restriction = jax.random.choice(
            sub_key_2, self.padded_dim, (self.sketch_size,), replace=True
        )

    def as_matrix(self) -> Array:
        # pad sign flip diagonal matrix with zero rows if needed
        S = jnp.vstack(
            [
                jnp.diag(self.sign_flip),
                jnp.zeros((self.padded_dim - self.dim, self.dim)),
            ]
        )
        # compute Hadamard transform
        S = 1.0 / jnp.sqrt(self.padded_dim) * hadamard_transform(self.padded_dim, S)
        # multiply with random restriction
        S = jnp.take(S, self.restriction, axis=0, unique_indices=False)
        return S

    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        # check dimension
        if (self.mode == "left") and (matrix.out_size() != self.dim):
            raise ValueError(
                "Output dimension of the linear operator does not match the input dimension of the sketching matrix."
            )
        if (self.mode == "right") and (matrix.in_size() != self.dim):
            raise ValueError(
                "Input dimension of the linear operator does not match the output dimension of the sketching matrix."
            )
        # generate new sketching matrix if needed
        if new_sketch:
            self._generate_sketching()
        # return composed operator
        if self.mode == "left":
            SA = jnp.vstack(
                [
                    jnp.multiply(self.sign_flip[:, None], matrix.as_matrix()),
                    jnp.zeros((self.padded_dim - self.dim, matrix.in_size())),
                ]
            )
            SA = (
                1.0
                / jnp.sqrt(self.padded_dim)
                * hadamard_transform(self.padded_dim, SA)
            )
            SA = jnp.take(SA, self.restriction, axis=0, unique_indices=False)
            return MatrixLinearOperator(SA) @ IdentityLinearOperator(
                jax.ShapeDtypeStruct(
                    shape=(matrix.in_size(),), dtype=default_floating_dtype()
                ),
                matrix.in_structure(),
            )
        else:
            raise ValueError("Right-sketching with SRHT is not supported.")

    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        # check mode
        if self.mode == "right":
            raise ValueError(
                "Sketch a vector with a right-sketching matrix is not allowed."
            )
        # ravel input vector
        unraveled, _ = ravel_tree(vector)
        # check dimension
        if len(unraveled) != self.dim:
            raise ValueError(
                "Vector length does not match the input dimension of the sketching matrix."
            )
        if new_sketch:
            self._generate_sketching()
        # compute sketched vector
        v = jnp.zeros(self.padded_dim)
        v = v.at[: self.dim].set(self.sign_flip * unraveled)
        v = (
            1.0
            / jnp.sqrt(self.padded_dim)
            * hadamard_transform(self.padded_dim, v.reshape(-1, 1)).flatten()
        )
        v = jnp.take(v, self.restriction, unique_indices=False)
        return v


class LESS(Sketching):
    r"""Leverage score sparsified embedding sketching matrix.

    Suppose the matrix to be sketched is of size :math:`\text{dim} \times m` with
    leverage scores :math:`(l_1, \cdots, l_{\text{dim}})`. Let :math:`s_i
    \sim \mathrm{Multinomial}(m, p_1, \cdots, p_{\text{dim}})` for
    :math:`i = 1, \cdots, \text{sketch\_size}` be i.i.d. multinomial random variables
    with probability :math:`p_j` approximately proportional to :math:`l_j` for :math:`j
    = 1, \cdots, \text{dim}`.

    Each row of the sketching matrix is independently determined by:

    .. math::
        s_i^{\mathsf{T}} = \sqrt{\frac{1}{\text{sketch\_size}}}
        \bigg(
        \sqrt{\frac{s_{i,1}}{p_1}} x_{i,1},
        \cdots,
        \sqrt{\frac{s_{i,\text{dim}}}{p_{\text{dim}}}} x_{i,\text{dim}}
        \bigg)

    where :math:`x_{i,j}`'s are i.i.d. mean zero and unit variance sub-Gaussian random
    variables (the implementation uses Rademacher random variables).

    The sketching has two additional optional arguments:

    - ``compute_leverage_scores`` specifies the approach to construct LESS embedding
      (default ``False``). If set to ``True``, then leverage scores are approximated
      [#f7]_. Otherwise the matrix gets transformed first so that leverage scores are
      approximately uniform.
    - ``matrix`` specifies the matrix to be sketched (default ``None``). If not provided,
      the sketching matrix construction is deferred when ``matrix_sketch`` gets called.

    References:
        .. [#f7]  P. Drineas, M. Magdon-Ismail, M. W. Mahoney, D. P. Woodruff, `Fast
          Approximation of Matrix Coherence and Statistical Leverage
          <https://jmlr.org/papers/v13/drineas12a.html>`_, Journal of Machine Learning
          Research, 13(111):3475–3506, 2012.

        - \M. Dereziński, Z. Liao, E. Dobriban, and M. W. Mahoney, `Sparse sketches with
          small inversion bias <https://proceedings.mlr.press/v134/derezinski21a>`_,
          Proceedings of 34\ :sup:`th` Conference on Learning Theory, PMLR 134:1467–1510
          , 2021.
    """

    def __init__(
        self,
        sketch_size: int,
        dim: int,
        seed: int,
        mode: str = "left",
        compute_leverage_scores=False,
        matrix=None,
    ):
        if mode == "right":
            raise ValueError("Right-sketching with LESS is not supported.")
        self.compute_leverage_scores = compute_leverage_scores
        # matrix is not needed with the data oblivious approach
        if self.compute_leverage_scores:
            self.matrix = matrix
        else:
            self.padded_dim = 2 ** np.ceil(np.log2(dim)).astype(np.int32)
        super().__init__(sketch_size, dim, seed, mode)
        # defer sketching if matrix is not provided at initialization
        if matrix is not None:
            self.m = matrix.in_size()
            self._generate_sketching()

    def _generate_sketching(self) -> None:
        if not hasattr(self, "m"):
            raise ValueError(
                "Matrix input dimension is not provided. Please initialize LESS embedding with a matrix or call matrix_sketch first."
            )
        # split keys
        self.key, sub_key_1, sub_key_2, sub_key_3 = jax.random.split(self.key, num=4)
        # use the leverage score approximation approach
        if self.compute_leverage_scores:
            if self.matrix is None:
                raise ValueError(
                    "Matrix to be sketched is not provided. Please initialize LESS embedding with a matrix or call matrix_sketch first."
                )
            # perform SVD of the sketched matrix
            SA = SJLT(
                self.m * 5, self.dim, int(jax.random.bits(sub_key_1))
            ).matrix_sketch(self.matrix)
            _, svals, rsvecs = jnp.linalg.svd(SA.as_matrix())
            # compute approximate orthogonal basis of the matrix
            AR_inv = self.matrix.as_matrix() @ rsvecs.T / svals.reshape(1, -1)
            # approximate leverage scores using squared row-norms of the orthogonal matrix
            lev_scores = jnp.sum(AR_inv**2, axis=1)
            # compute sampling probability according to approximate leverage scores
            probs = lev_scores / jnp.sum(lev_scores)
            # draw multinomial random variables
            # JAX random module does not support multinomial yet
            # we use less performant workaround for now
            # see https://github.com/jax-ml/jax/issues/13327 for a discussion on this
            cat_samples = jax.random.categorical(
                sub_key_2, logits=jnp.log(probs), shape=(self.sketch_size, self.m)
            )
            bincount_1d_fun = lambda M: jnp.bincount(M, length=self.dim)
            samples = jax.vmap(bincount_1d_fun, 0)(cat_samples)
            samples = samples / (self.m * probs)
            # assemble LESS embedding
            self.sketching = jnp.sqrt(
                samples / self.sketch_size
            ) * jax.random.rademacher(sub_key_3, (self.sketch_size, self.dim))
        # use the pre-processing approach that uniformizes the leverage scores
        else:
            # generate random sign flips for the Hadamard transform
            self.sign_flip = jax.random.rademacher(sub_key_1, (self.dim,))
            # use uniformly sparsified sketching matrix
            cat_samples = jax.random.categorical(
                sub_key_2,
                logits=-jnp.log(self.padded_dim) * jnp.ones(self.padded_dim),
                shape=(self.sketch_size, self.m),
            )
            bincount_1d_fun = lambda M: jnp.bincount(M, length=self.padded_dim)
            samples = jax.vmap(bincount_1d_fun, 0)(cat_samples)
            samples = samples / (self.m / self.padded_dim)
            # assemble LESS embedding
            self.sketching = jnp.sqrt(
                samples / self.sketch_size
            ) * jax.random.rademacher(sub_key_3, (self.sketch_size, self.padded_dim))

    def as_matrix(self) -> Array:
        if self.compute_leverage_scores:
            return self.sketching
        else:
            return (
                1.0
                / jnp.sqrt(self.padded_dim)
                * self.sketching
                @ hadamard_transform(
                    self.padded_dim,
                    jnp.vstack(
                        [
                            jnp.diag(self.sign_flip),
                            jnp.zeros((self.padded_dim - self.dim, self.dim)),
                        ]
                    ),
                )
            )

    def matrix_sketch(
        self, matrix: AbstractLinearOperator, new_sketch=False
    ) -> AbstractLinearOperator:
        # check dimension
        if (self.mode == "left") and (matrix.out_size() != self.dim):
            raise ValueError(
                "Output dimension of the linear operator does not match the input dimension of the sketching matrix."
            )
        if (self.mode == "right") and (matrix.in_size() != self.dim):
            raise ValueError(
                "Input dimension of the linear operator does not match the output dimension of the sketching matrix."
            )
        # generate new sketching matrix if needed
        if new_sketch or not hasattr(self, "sketching"):
            self.m = matrix.in_size()
            if self.compute_leverage_scores:
                # set new matrix
                self.matrix = matrix
            self._generate_sketching()
        # return composed operator
        if self.mode == "left":
            if self.compute_leverage_scores:
                return MatrixLinearOperator(
                    self.sketching @ matrix.as_matrix()
                ) @ IdentityLinearOperator(
                    jax.ShapeDtypeStruct(
                        shape=(matrix.in_size(),), dtype=default_floating_dtype()
                    ),
                    matrix.in_structure(),
                )
            else:
                # transform the matrix to make leverage scores approximately uniform
                SA = jnp.vstack(
                    [
                        jnp.multiply(self.sign_flip[:, None], matrix.as_matrix()),
                        jnp.zeros((self.padded_dim - self.dim, matrix.in_size())),
                    ]
                )
                SA = (
                    1.0
                    / jnp.sqrt(self.padded_dim)
                    * hadamard_transform(self.padded_dim, SA)
                )
                # apply sketching to the transformed matrix
                return MatrixLinearOperator(
                    self.sketching @ SA
                ) @ IdentityLinearOperator(
                    jax.ShapeDtypeStruct(
                        shape=(matrix.in_size(),), dtype=default_floating_dtype()
                    ),
                    matrix.in_structure(),
                )
        else:
            raise ValueError("Right-sketching with LESS is not supported.")

    def vector_sketch(self, vector: Any, new_sketch=False) -> Array:
        # check mode
        if self.mode == "right":
            raise ValueError(
                "Sketch a vector with a right-sketching matrix is not allowed."
            )
        # ravel input vector
        unraveled, _ = ravel_tree(vector)
        # check dimension
        if len(unraveled) != self.dim:
            raise ValueError(
                "Vector length does not match the input dimension of the sketching matrix."
            )
        if new_sketch or not hasattr(self, "sketching"):
            self._generate_sketching()
        # compute sketched vector
        if self.compute_leverage_scores:
            return self.sketching @ unraveled
        else:
            v = jnp.zeros(self.padded_dim)
            v = v.at[: self.dim].set(self.sign_flip * unraveled)
            v = (
                1.0
                / jnp.sqrt(self.padded_dim)
                * hadamard_transform(self.padded_dim, v.reshape(-1, 1)).flatten()
            )
            return self.sketching @ v


class UniformRowSampling(Sketching):
    r"""Uniform row sampling sketching matrix.

    References:
        - \M. B. Cohen, Y. T. Lee, C. [Cameron] Musco, C. [Christopher] Musco, R. Peng,
          A. Sidford, `Uniform Sampling for Matrix Approximation
          <https://doi.org/10.1145/2688073.2688113>`_, Proceedings of the 2015
          Conference on Innovations in Theoretical Computer Science, Association for
          Computing Machinery 181–190, 2015.
    """


class RowNormSampling(Sketching):
    r"""Row-norm sampling sketching matrix."""


# Mapping of sketch types to sketching classes
SKETCH_TYPE_OPTIONS = {
    "gaussian": GaussianEmbedding,
    "orthonormal": OrthonormalEmbedding,
    "sjlt": SJLT,
    "srtt": SRTT,
    "srht": SRHT,
    "less": LESS,
    "row": UniformRowSampling,
    "row_norm": RowNormSampling,
}
