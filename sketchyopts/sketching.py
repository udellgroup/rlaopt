import abc

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from lineax import AbstractLinearOperator

from sketchyopts.util import (
    default_floating_dtype,
    form_dense_vector,
    frozenset,
    ravel_tree,
    sample_indices,
    tree_size,
)


class RandomizedSketching(AbstractLinearOperator):
    r"""Abstract class for randomized sketching.

    The class defines the interface for a randomized sketching operator. The class
    represents a sketched version of a linear operator.
    """

    def __init__(self, operator, sketch_size, key, tags=()):
        r"""Initialize the sketched operator.

        Args:
          operator: Linear operator to sketch.
          sketch_size: Size of the sketch.
          key: Random key.
          tags: Tags for the sketched operator. See `Lineax Tags
            <https://docs.kidger.site/lineax/api/tags/>`_ for more details.
        """
        self.operator = operator
        self.sketch_size = sketch_size
        self.key = key
        self.tags = frozenset(tags)

    @abc.abstractmethod
    def generate_sketching(self) -> None:
        r"""Generate the sketching.

        The method generates the sketching internally. The method is called when the
        sketching is not provided during the initialization. Users can also call the
        method to re-generate the sketching.
        """

    @abc.abstractmethod
    def apply_sketch(self, vector) -> Array:
        r"""Apply the sketching to a vector.

        The method applies the sketching direcrly to a vector of the compatible size
        (*i.e.* the size of the operator's output). This differs from the matrix-vector
        product, which applies the sketched operator to a vector of the size of the
        operator's input.

        Mathematically if we denote the operator as :math:`A` and the sketching matrix
        as :math:`S`, the method computes :math:`S x` whereas the matrix-vector product
        computes :math:`S A x`.

        Args:
          vector: Vector that the sketching applies to. The vector should be a pytree
            of floating-point arrays whose structure matches the output structure of
            the operator.

        Returns:
          \Sketched vector. The output is a floating-point array of length equal to the
          sketch size.
        """
        pass

    def mv(self, vector):
        r"""Apply the sketched operator to a vector.

        The method computes the matrix-vector product between the sketched operator and
        a vector.

        Args:
          vector: Vector that the sketched operator applies to. The vector should be a
            pytree of floating-point arrays whose structure matches the input
            structure of the operator.

        Returns:
          \Sketched vector. The output is a floating-point array of length equal to the
          sketch size.

        See Also:
          :func:`~sketchyopts.sketching.RandomizedSketching.apply_sketch`
        """
        return self.apply_sketch(self.operator.mv(vector))

    @abc.abstractmethod
    def as_matrix(self) -> Array:
        r"""Materialize the sketched operator.

        The method materializes the sketched operator as a matrix. Since many linear
        operators are defined implicitly and sketching is not necessarily stored as a
        full matrix, materializing can be a computationally expensive operation.

        Returns:
          \Sketched operator as a 2D JAX array. The output is a floating-point array of
          shape :math:`(\text{{sketch\_size}}, \text{{operator\_input\_size}})`.
        """

    def transpose(self):
        r"""Transpose the sketched operator.

        The randomized sketching class currently does not support transposition.
        """
        raise NotImplementedError

    def in_structure(self):
        r"""Return the input structure of the sketched operator.

        Returns:
          Input structure of the sketched operator which is the input structure of the
          operator.
        """
        return self.operator.in_structure()

    def out_structure(self):
        r"""Return the output structure of the sketched operator.

        Returns:
          Output structure of the sketched operator which is an array of length equal to
          the sketch size.
        """
        return jax.ShapeDtypeStruct(
            shape=(self.sketch_size,), dtype=default_floating_dtype()
        )

    def in_size(self):
        r"""Return the total number of scalars in the input of the sketched operator.

        Returns:
          An integer representing the dimensionality of the input space.
        """
        return tree_size(self.in_structure())

    def out_size(self):
        r"""Return the total number of scalars in the output of the sketched operator.

        Returns:
          An integer representing the dimensionality of the output space.
        """
        return tree_size(self.out_structure())


class GaussianEmbedding(RandomizedSketching):
    r"""Gaussian embedding sketching operator.

    The class defines a Gaussian embedding sketching operator. The sketching is
    defined by a Gaussian random matrix (each entry is independently drawn from a
    normal distribution with zero mean and variance :math:`1 / \text{sketch\_size}`).

    Args:
      operator: Linear operator to sketch.
      sketch_size: Size of the sketch.
      key: Random key.
      tags: Tags for the sketched operator (default ``None``).
      sketch_matrix: Sketching matrix (default ``None``). If provided, the operator uses
        the provided matrix for sketching. Otherwise, the operator generates the
        sketching matrix at initialization.
    """

    def __init__(self, operator, sketch_size, key, *, tags=(), sketch_matrix=None):
        r"""Initialize the Gaussian embedding sketching operator."""
        super().__init__(operator, sketch_size, key, tags)
        # generate sketching if needed
        if sketch_matrix is not None:
            self.matrix = sketch_matrix
        else:
            self.generate_sketching()

    def generate_sketching(self):
        r"""Generate the sketching matrix."""
        self.key, sub_key = jax.random.split(self.key)
        self.sketch_matrix = (1.0 / jnp.sqrt(self.sketch_size)) * jax.random.normal(
            sub_key, (self.sketch_size, self.operator.out_size())
        )

    def apply_sketch(self, vector):
        r"""Apply the sketching to a vector."""
        return jnp.matmul(
            self.matrix, ravel_tree(vector)[0], precision=jax.lax.Precision.HIGHEST
        )

    def as_matrix(self):
        r"""Materialize the sketched operator."""
        return jnp.matmul(
            self.matrix, self.operator.as_matrix(), precision=jax.lax.Precision.HIGHEST
        )


class SRTT(RandomizedSketching):
    r"""Subsampled randomized trigonometric transforms (SRTT) sketching operator.

    The class defines a subsampled randomized trigonometric transforms (SRTT) sketching
    operator. The sketching :math:`S` is defined by a random restriction :math:`R`, a
    discrete cosine transform :math:`F`, a random sign flip :math:`E`, and a random
    permutation :math:`\Gamma`:

    .. math::
        S = \sqrt{\frac{n}{d}} E F R \Gamma

    where :math:`n` is the size of the operator's output, :math:`d` is the size of the
    sketch.

    If the permutation, sign flip, and restriction arrays are provided, the operator
    uses the provided arrays for sketching. Otherwise, the operator generates the
    matrices at initialization.

    Args:
      operator: Linear operator to sketch.
      sketch_size: Size of the sketch.
      key: Random key.
      tags: Tags for the sketched operator (default ``None``).
      permutation: Permutation array (default ``None``). It is stored as a vector of
        permutation of the operator's output indices.
      sign_flip: Sign flip array (default ``None``). It is stored as a vector of random
        signs independently drawn uniform :math:`\{\pm 1\}`) of size of the operator's
        output.
      restriction: Restriction array (default ``None``). It is stored as a vector of
        randomly drawn indices of size of the sketch from the operator's output indices.
    """

    def __init__(
        self,
        operator,
        sketch_size,
        key,
        *,
        tags=(),
        permutation=None,
        sign_flip=None,
        restriction=None,
    ):
        r"""Initialize the SRTT sketching operator."""
        super().__init__(operator, sketch_size, key, tags)
        # generate sketching if needed
        if (
            (permutation is not None)
            and (sign_flip is not None)
            and (restriction is not None)
        ):
            self.permutation = permutation
            self.sign_flip = sign_flip
            self.restriction = restriction
        else:
            self.generate_sketching()

    def generate_sketching(self):
        r"""Generate the sketching matrices."""
        # split keys
        self.key, sub_key_1, sub_key_2, sub_key_3 = jax.random.split(self.key, num=4)
        # generate sketching components
        self.permutation = jax.random.permutation(sub_key_1, self.operator.out_size())
        self.sign_flip = (
            jax.random.randint(sub_key_2, self.operator.out_size(), minval=0, maxval=2)
            * 2
            - 1
        )
        self.restriction = jax.random.choice(
            sub_key_3, self.operator.out_size(), (self.sketch_size,), replace=False
        )

    def apply_sketch(self, vector):
        r"""Apply the sketching to a vector."""
        v = jnp.put(
            jnp.zeros(self.operator.out_size()), self.permutation, ravel_tree(vector)[0]
        )
        v = self.sign_flip * v
        v = jsp.fft.dct(v)
        v = jnp.take(v, self.restriction, unique_indices=True)
        return jnp.sqrt(self.operator.out_size() / self.sketch_size) * v

    def as_matrix(self):
        r"""Materialize the sketched operator."""
        M = jax.numpy.take(
            self.operator.as_matrix(), self.permutation, axis=0, unique_indices=True
        )
        M = jax.numpy.multiply(self.sign_flip[:, None], M)
        M = jsp.fft.dct(M, axis=0)
        M = jnp.take(M, self.restriction, axis=0, unique_indices=True)
        return jnp.sqrt(self.operator.out_size() / self.sketch_size) * M


class SparseSignEmbedding(RandomizedSketching):
    r"""Sparse sign embedding sketching operator.

    The class defines a sparse sign embedding sketching operator. The sketching
    :math:`S` is defined by a sparse matrix where each column has exactly :math:`\zeta`
    non-zero entries. Each non-zero entry is independently drawn uniformly from
    :math:`\{\pm 1\}`, and its location is randomly chosen from :math:`\{1, \cdots, d\}`
    where :math:`d` is the size of the sketch. Mathematically, the sketching takes the
    form:

    .. math::
        S = \sqrt{\frac{1}{\xi}} \big[s_1, \cdots, s_n\big]

    Here :math:`n` is the size of the operator's output.

    If the non-zero indices and corresponding random signs are both provided, the
    operator uses the provided arrays for sketching. Otherwise, the operator generates
    the matrices at initialization.

    The sparsity parameter :math:`\zeta` takes the recommended practical value of
    :math:`\operatorname{min}\{8, d\}` [#f6]_.

    .. rubric:: References

    .. [#f6]  J\. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, `Streaming Low-Rank
      Matrix Approximation with an Application to Scientific Simulation <https://epubs.
      siam.org/doi/10.1137/18M1201068>`_, SIAM Journal on Scientific Computing, 41(4),
      2019.

    Args:
      operator: Linear operator to sketch.
      sketch_size: Size of the sketch.
      key: Random key.
      tags: Tags for the sketched operator (default ``None``).
      sparsity: Sparsity parameter :math:`\zeta` (default ``None``). If provided, the
        operator uses the provided value for the sparsity. Otherwise, the operator uses
        the recommended value of :math:`\operatorname{min}\{8, d\}`.
      coord_matrix: Coordinate matrix (default ``None``).
      sign_matrix: Sign matrix (default ``None``).
    """

    def __init__(
        self,
        operator,
        sketch_size,
        key,
        *,
        tags=(),
        sparsity=None,
        coord_matrix=None,
        sign_matrix=None,
    ):
        r"""Initialize the sparse sign embedding sketching operator."""
        super().__init__(operator, sketch_size, key, tags)
        # set sparsity
        if sparsity is not None:
            self.sparsity = sparsity
        else:
            self.sparsity = min(8, self.sketch_size)
        # generate sketching if needed
        if (coord_matrix is not None) and (sign_matrix is not None):
            self.coord_matrix = coord_matrix
            self.sign_matrix = sign_matrix
        else:
            self.generate_sketching()

    def generate_sketching(self):
        r"""Generate the sketching matrices."""
        # split keys
        self.key, sub_key_1, sub_key_2 = jax.random.split(self.key, num=3)
        # generate random coordinates for non-zero entries
        key_array = jax.random.split(sub_key_1, num=self.operator.out_size())
        self.coord_matrix = jax.vmap(sample_indices, in_axes=(None, None, 0))(
            self.operator.out_size(), self.sparsity, key_array
        )
        # generate random signs
        self.sign_matrix = (
            jax.random.randint(
                sub_key_2, (self.sparsity, self.operator.out_size()), minval=0, maxval=2
            )
            * 2
            - 1
        )

    def apply_sketch(self, vector):
        r"""Apply the sketching to a vector."""
        return jnp.sqrt(1.0 / self.sparsity) * jnp.cumsum(
            jax.vmap(form_dense_vector, in_axes=(1, 1, None, 0))(
                self.coord_matrix,
                self.sign_matrix,
                self.sketch_size,
                ravel_tree(vector)[0],
            ),
            axis=1,
        )

    def as_matrix(self):
        r"""Materialize the sketched operator."""
        return jnp.apply_over_axes(self.apply_sketch, self.operator.as_matrix(), [1])


# Mapping of sketch types to sketching classes
SKETCH_TYPE_OPTIONS = {
    "gaussian": GaussianEmbedding,
    "srft": SRTT,
    "sparse-sign": SparseSignEmbedding,
}
