import jax
import jax.numpy as jnp
import pytest

from sketchyopts import errors, solver
from tests.test_util import TestCase


class TestNystromPCG(TestCase):

    key = jax.random.PRNGKey(0)

    def test_nystrom_pcg_correctness_1(self):
        """
        Test a trivial linear system with a single righthand side.

        The matrix is identity and we expect the solution to match the generated
        righthand side.
        """
        size = 10
        mu = 0
        rank = 1

        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        A = jnp.identity(size)
        b = jax.random.normal(subkey1, (size, 1))
        x, state = solver.nystrom_pcg(A, b, mu, rank, subkey2)

        self.assertAllClose(x, b)
        assert state.iter_num < size * 10

    # def test_nystrom_pcg_multiple_righthand_sides(self):
    #     """
    #     Test a trivial linear system with multiple righthand sides.

    #     The matrix is identity and we expect the solution to match the generated
    #     righthand sides.
    #     """
    #     size = 10
    #     num_rhs = 20
    #     mu = 0
    #     rank = 1

    #     self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
    #     A = jnp.identity(size)
    #     b = jax.random.normal(subkey1, (size, num_rhs))
    #     x, _, status, k = solver.nystrom_pcg(A, b, mu, rank, subkey2)

    #     self.assertAllClose(x, b)
    #     self.assertAllClose(status, jnp.ones(num_rhs, dtype=bool))
    #     assert k < size * 10

    def test_nystrom_pcg_correctness_2(self):
        """
        Test a more realistic linear system with single righthand side.

        The matrix has exponential spectral decay (hence ill-conditioned). We repeat the
        solve with different keys and compute the norm of the different between
        approximate solution and the true solution for each key.

        Based on the tolerance level set for the residual, we expect the approximate
        solutions to be close as well.
        """
        size = 100
        mu = 0.1
        rank = 10
        num_repeats = 10
        tolerance = 1e-05

        # generate PSD matrix with fast spectral decay
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        Q, _ = jnp.linalg.qr(jax.random.normal(subkey1, (size, size)))
        L = jnp.exp(1 / (jnp.arange(size) + 0.5)) - 1
        A = Q @ jnp.diag(L) @ Q.T
        x = jax.random.normal(subkey2, (size, 1))
        b = A @ x + mu * x

        # run Nyström-PCG
        tol_factor = 1 / (L[-1] ** 2 + 2 * L[-1] * mu + mu**2) ** (1 / 2)
        for i in range(num_repeats):
            self.key, subkey = jax.random.split(self.key)
            x_final, _ = solver.nystrom_pcg(A, b, mu, rank, subkey, tol=tolerance)
            # approximate solution should be close to the true solution
            assert jnp.linalg.norm(x - x_final) <= tol_factor * tolerance

    # def test_nystrom_pcg_errors(self):
    #     """
    #     Test various input errors.
    #     """
    #     rank = 1
    #     mu = 1

    #     # wrong dimension
    #     A = 0
    #     b = 0
    #     self.key, subkey = jax.random.split(self.key)
    #     with pytest.raises(
    #         errors.InputDimError,
    #         match="Input A is expected to have dimension 2 but has 0.",
    #     ):
    #         solver.nystrom_pcg(A, b, mu, rank, subkey)

    #     A = jnp.ones(10)
    #     self.key, subkey = jax.random.split(self.key)
    #     with pytest.raises(
    #         errors.InputDimError,
    #         match="Input A is expected to have dimension 2 but has 1.",
    #     ):
    #         solver.nystrom_pcg(A, b, mu, rank, subkey)

    #     A = jnp.ones((10, 10, 10))
    #     self.key, subkey = jax.random.split(self.key)
    #     with pytest.raises(
    #         errors.InputDimError,
    #         match="Input A is expected to have dimension 2 but has 3.",
    #     ):
    #         solver.nystrom_pcg(A, b, mu, rank, subkey)

    #     A = jnp.ones((10, 10))
    #     self.key, subkey = jax.random.split(self.key)
    #     with pytest.raises(
    #         errors.InputDimError,
    #         match="Input b is expected to have any dimension in \\[1, 2\\] but has 0.",
    #     ):
    #         solver.nystrom_pcg(A, b, mu, rank, subkey)

    #     b = jnp.ones((10, 2, 1))
    #     self.key, subkey = jax.random.split(self.key)
    #     with pytest.raises(
    #         errors.InputDimError,
    #         match="Input b is expected to have any dimension in \\[1, 2\\] but has 3.",
    #     ):
    #         solver.nystrom_pcg(A, b, mu, rank, subkey)

    #     # wrong shape
    #     A = jnp.ones((10, 5))
    #     self.key, subkey = jax.random.split(self.key)
    #     with pytest.raises(
    #         errors.MatrixNotSquareError,
    #         match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
    #     ):
    #         solver.nystrom_pcg(A, b, mu, rank, subkey)
