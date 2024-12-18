import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from scipy.linalg import hadamard

from sketchyopts import sketching
from tests.test_util import TestCase


class TestHadamard(TestCase):

    def test_hadamard_transform(self):
        seed = 42
        dim = 8  # Hadamard transform requires dimensions to be a power of 2
        m = 10
        matrix = jax.random.normal(jax.random.PRNGKey(seed), (dim, m))

        # compute the transformed vector using our implementation
        transformed_vector = sketching.hadamard_transform(dim, matrix)

        # compute the expected result using scipy's Hadamard transform
        hadamard_matrix = hadamard(dim)
        expected_transformed_vector = jnp.matmul(hadamard_matrix, matrix)

        # assert equality
        self.assertAllClose(transformed_vector, expected_transformed_vector, rtol=1e-5)


class TestSketchingMatrices(TestCase):

    @pytest.mark.parametrize(
        "sketch_class",
        [
            (sketching.GaussianEmbedding, {}),
            (sketching.OrthonormalEmbedding, {}),
            (sketching.SJLT, {}),
            (sketching.SRTT, {}),
            (sketching.SRHT, {}),
            (sketching.LESS, {"compute_leverage_scores": True}),
            (sketching.LESS, {"compute_leverage_scores": False}),
        ],
    )
    # @pytest.mark.parametrize("sketch_class", [(sketching.SJLT, {})])
    def test_sketching_matrices(self, sketch_class):
        seed = 42
        sketch_size = 100
        out_dim = 1000
        in_dim = 100
        num_rand_vectors = 1000
        key = jax.random.PRNGKey(seed)

        # create an instance of the sketching class
        sketch_name, sketch_kwargs = sketch_class
        sketch = sketch_name(sketch_size, out_dim, seed, mode="left", **sketch_kwargs)

        # generate a random matrix
        key, sub_key = jax.random.split(key)
        matrix = jax.random.normal(sub_key, (out_dim, in_dim))

        # compute the sketched matrix
        sketched_matrix = sketch.matrix_sketch(
            lx.MatrixLinearOperator(matrix)
        ).as_matrix()

        # verify matrix instantiation and vector sketching are correct
        if sketch_name != sketching.SRTT:
            key, sub_key = jax.random.split(key)
            vector = jax.random.normal(sub_key, (out_dim,))
            self.assertAllClose(
                sketched_matrix,
                jnp.matmul(
                    sketch.as_matrix(), matrix, precision=jax.lax.Precision.HIGHEST
                ),
                atol=1e-5,
            ), f"Matrix sketching is incorrect for {sketch_class}"
            self.assertAllClose(
                sketch.vector_sketch(vector),
                jnp.matmul(
                    sketch.as_matrix(), vector, precision=jax.lax.Precision.HIGHEST
                ),
                atol=1e-5,
            ), f"Vector sketching is incorrect for {sketch_class}"

        # generate random vectors
        key, sub_key = jax.random.split(key)
        vectors = jax.random.normal(sub_key, (in_dim, num_rand_vectors))

        # column space preservation by subspace embedding
        projected_vectors = jnp.matmul(matrix, vectors)
        sketched_vectors = jnp.matmul(sketched_matrix, vectors)
        norm_projected_vectors = jnp.linalg.norm(projected_vectors, axis=0)
        norm_sketched_vectors = jnp.linalg.norm(sketched_vectors, axis=0)
        relative_error_distances = jnp.linalg.norm(
            norm_projected_vectors - norm_sketched_vectors
        ) / jnp.linalg.norm(norm_projected_vectors)
        assert (
            relative_error_distances < 1e-0
        ), f"Relative error in subspace preservation too high: {relative_error_distances}"
