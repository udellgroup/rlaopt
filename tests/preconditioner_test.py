import jax.random
import jax.numpy as jnp
import pytest

from sketchyopts.preconditioner import (
    rand_nystrom_approx,
    update_nystrom_precond,
    scale_by_nystrom_precond,
)
from sketchyopts.util import shareble_state_named_chain
from sketchyopts.errors import InputDimError, MatrixNotSquareError
from optax._src.base import GradientTransformation
from collections import namedtuple


class TestNystromApprox:

    def test_rand_nystrom_approx_correctness(self):
        """
        Test the approximation quality based on element-wise relative error.
        We generate the original matrix from Normal distirbution and repeatedly
        approximate the matrix with different keys. We then compute the average
        element-wise relative error from these runs.
        """
        size = 100
        rank = 5
        approx_rank = 10
        seed = 0
        num_repeats = 100
        tolerance = 1e-05
        # generate keys
        initial_key = jax.random.PRNGKey(seed)
        generation_key, approximation_key = jax.random.split(initial_key)
        # generate PSD matrix
        vecs = jax.random.normal(generation_key, (size, rank))
        A = vecs @ vecs.T
        # compute randomized Nyström approximations
        key = approximation_key
        mean_relative_error = 0.0
        for i in range(num_repeats):
            key, subkey = jax.random.split(key)
            U, S = rand_nystrom_approx(A, approx_rank, subkey)
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
        key = jax.random.PRNGKey(0)

        # wrong dimension
        A = 0
        with pytest.raises(
            InputDimError, match="Input A is expected to have dimension 2 but has 0."
        ):
            rand_nystrom_approx(A, rank, key)

        A = jnp.ones(10)
        with pytest.raises(
            InputDimError, match="Input A is expected to have dimension 2 but has 1."
        ):
            rand_nystrom_approx(A, rank, key)

        A = jnp.ones((10, 10, 10))
        with pytest.raises(
            InputDimError, match="Input A is expected to have dimension 2 but has 3."
        ):
            rand_nystrom_approx(A, rank, key)

        # wrong shape
        A = A = jnp.ones((10, 5))
        with pytest.raises(
            MatrixNotSquareError,
            match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
        ):
            rand_nystrom_approx(A, rank, key)


class TestNystromPrecond:

    @pytest.fixture(scope="class")
    def test_parameters(self):
        """
        Define test parameters.
        """
        size = 10
        rho = 1.0
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, num=3)
        v = jax.random.normal(key1, (size,))
        # generate eigen-decomposition of a symmetric positive-definite matrix
        U, _ = jnp.linalg.qr(jax.random.normal(key2, (size, size)))
        S = jnp.sort(jnp.abs(jax.random.normal(key3, (size,))), descending=True)

        return namedtuple("TestParameters", ["size", "rho", "v", "U", "S"])(
            size, rho, v, U, S
        )

    @pytest.fixture(scope="class")
    def pass_precond_transformation(self):
        """
        Define a transformation that stores eigen-decomposition of a preconditioner.
        """

        def pass_precond(U, S):
            def init_fn(params):
                del params
                return namedtuple("PrecondState", ["U", "S"])(U, S)

            def update_fn(updates, state, params=None):
                del params
                return updates, state

            return GradientTransformation(init_fn, update_fn)

        return pass_precond

    @pytest.fixture(scope="class")
    def test_benchmarks(self, test_parameters):
        """
        Compute the expected preconditioned vector and learning rate.
        """
        precond_v = (
            test_parameters.U
            @ jnp.diag(1.0 / (test_parameters.S + test_parameters.rho))
            @ test_parameters.U.T
        ) @ test_parameters.v
        lr = 1.0 / jnp.max(
            test_parameters.S / (test_parameters.S + test_parameters.rho)
        )

        return namedtuple("TestBenchmarks", ["precond_v", "lr"])(precond_v, lr)

    @pytest.fixture(scope="class")
    def objective_function(self, test_parameters):
        """
        Define an objective function whose Hessian matches the provided eigen-decomposition.
        """
        A = (
            test_parameters.U
            @ jnp.diag(jnp.sqrt(test_parameters.S))
            @ test_parameters.U.T
        )

        def f(x):
            return (1 / 2) * jnp.sum(jnp.square(A @ x))

        return f

    def test_update_nystrom_precond_construction(
        self, test_parameters, test_benchmarks, objective_function
    ):
        """
        Test preconditioner construction.
        """
        params = jnp.ones(test_parameters.size)
        tx = update_nystrom_precond(
            test_parameters.size, test_parameters.rho, 0, 0, objective_function
        )
        state = tx.init(params)
        _, state = tx.update(params, state, params)

        h = test_parameters.U @ jnp.diag(test_parameters.S) @ test_parameters.U.T
        h_hat = state.U @ jnp.diag(state.S) @ state.U.T

        assert jnp.allclose(state.step_count, jnp.ones([]))
        assert jnp.allclose(state.S, test_parameters.S)
        assert jnp.allclose(h, h_hat, rtol=1e-02)
        assert jnp.allclose(state.learning_rate, test_benchmarks.lr, rtol=1e-02)

    def test_update_nystrom_precond_updates(self, test_parameters, test_benchmarks):
        """
        Test preconditioner updates.
        """
        f = lambda x: (1 / 12) * jnp.sum(jnp.power(x, 4))
        update_freq = 2
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)

        params_ones = jnp.ones(test_parameters.size)
        params_rand = jax.random.normal(key1, (test_parameters.size,))
        tx = update_nystrom_precond(
            test_parameters.size, test_parameters.rho, update_freq, 0, f
        )
        state = tx.init(params_ones)
        # first iteration: eigenvalues should be all ones
        _, state = tx.update(params_ones, state, params_ones)
        assert jnp.allclose(state.S, params_ones)
        # second iteration: eigenvalues remain all ones
        _, state = tx.update(params_ones, state, params_rand)
        assert jnp.allclose(state.S, params_ones)
        # third iteration: eigenvalues get updated
        _, state = tx.update(params_ones, state, params_rand)
        assert jnp.allclose(
            state.S, jnp.sort(jnp.square(params_rand), descending=True), rtol=1e-02
        )
        # fourth iteration: eigenvalues stay the same
        _, state = tx.update(params_ones, state, params_ones)
        assert jnp.allclose(
            state.S, jnp.sort(jnp.square(params_rand), descending=True), rtol=1e-02
        )

    def test_scale_by_nystrom_precond_specified_precond(
        self, test_parameters, test_benchmarks
    ):
        """
        Test preconditioning computation using directly provided preconditioner.
        We compare the update from the transformation against the manually computed result.
        """
        tx = scale_by_nystrom_precond(
            test_parameters.rho, U=test_parameters.U, S=test_parameters.S
        )
        state = tx.init(jnp.ones(test_parameters.size))
        update, state = tx.update(test_parameters.v, state)

        assert jnp.allclose(update, test_benchmarks.precond_v)

    def test_scale_by_nystrom_precond_ref_precond(
        self, test_parameters, test_benchmarks, pass_precond_transformation
    ):
        """
        Test preconditioning computation using referenced preconditioner.
        We compare the update from the transformation against the manually computed result.
        """
        chain = shareble_state_named_chain(
            (
                "precond",
                pass_precond_transformation(test_parameters.U, test_parameters.S),
            ),
            (
                "scale_by_nystrom",
                scale_by_nystrom_precond(test_parameters.rho, ref_state="precond"),
            ),
        )
        state = chain.init(jnp.ones(test_parameters.size))
        update, state = chain.update(test_parameters.v, state)

        assert jnp.allclose(update, test_benchmarks.precond_v)
