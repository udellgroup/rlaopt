import jax
import jax.numpy as jnp
import pytest

from sketchyopts import errors, nystrom
from tests.test_util import TestCase


class TestNystromApprox(TestCase):

    key = jax.random.PRNGKey(0)

    def test_rand_nystrom_approx_correctness(self):
        """
        Test the approximation quality based on element-wise relative error.

        We generate the original matrix from Normal distirbution and repeatedly
        approximate the matrix with different keys. We then compute the average
        element-wise relative error from these runs.
        """
        size = 100
        rank = 10
        num_repeats = 1000
        tolerance = 1e-03
        # generate keys
        self.key, generation_key, approximation_key = jax.random.split(self.key, num=3)
        # generate PSD matrix
        A = jax.random.uniform(generation_key, (rank, size))
        A = A.T @ A
        # jit the approximation function
        nys_approx = jax.jit(lambda a, k: nystrom.rand_nystrom_approx(a, rank, k))
        # compute randomized Nyström approximations
        mean_relative_error = 0.0
        for i in range(num_repeats):
            approximation_key, subkey = jax.random.split(approximation_key)
            U, S = nys_approx(A, subkey)
            A_nys = U @ jnp.diag(S) @ U.T
            # keep track of average element-wise relative errors
            mean_relative_error += (1 / num_repeats) * jnp.mean(
                jnp.absolute(A_nys - A) / jnp.absolute(A)
            )
        # approximated matrix should be close to the original one (on average)
        assert mean_relative_error <= tolerance

    def test_rand_nystrom_approx_errors(self):
        """
        Test various input errors.
        """
        rank = 1

        # wrong dimension
        A = 0
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input A is expected to have dimension 2 but has 0.",
        ):
            nystrom.rand_nystrom_approx(A, rank, subkey)

        A = jnp.ones(10)
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input A is expected to have dimension 2 but has 1.",
        ):
            nystrom.rand_nystrom_approx(A, rank, subkey)

        A = jnp.ones((10, 10, 10))
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input A is expected to have dimension 2 but has 3.",
        ):
            nystrom.rand_nystrom_approx(A, rank, subkey)

        # wrong shape
        A = jnp.ones((10, 5))
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.MatrixNotSquareError,
            match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
        ):
            nystrom.rand_nystrom_approx(A, rank, subkey)
