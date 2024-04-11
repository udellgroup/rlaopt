import jax.random
import jax.numpy as jnp
import optax
import pytest

from sketchyopts.solver import nystrom_pcg, sketchysgd
from sketchyopts.errors import InputDimError, MatrixNotSquareError
from collections import namedtuple


class TestNystromPCG:

    def test_nystrom_pcg_correctness_1(self):
        """
        Test a trivial linear system with a single righthand side.
        The matrix is identity and we expect the solution to match the generated righthand side.
        """
        size = 10
        mu = 0
        rank = 1
        seed = 0

        key = jax.random.PRNGKey(seed)
        A = jnp.identity(size)
        b = jax.random.normal(key, (size, 1))
        x, _, status, k = nystrom_pcg(A, b, mu, rank, key)
        assert jnp.allclose(x, b)
        assert jnp.allclose(status, True)

    def test_nystrom_pcg_multiple_righthand_sides(self):
        """
        Test a trivial linear system with multiple righthand sides.
        The matrix is identity and we expect the solution to match the generated righthand sides.
        """
        size = 10
        num_rhs = 20
        mu = 0
        rank = 1
        seed = 0

        key = jax.random.PRNGKey(seed)
        A = jnp.identity(size)
        b = jax.random.normal(key, (size, num_rhs))
        x, _, status, k = nystrom_pcg(A, b, mu, rank, key)
        print(k)

        assert jnp.allclose(x, b)
        assert jnp.allclose(status, jnp.ones(num_rhs))
        assert k < size * 10

    def test_nystrom_pcg_correctness_2(self):
        """
        Test a more realistic linear system with single righthand side.
        The matrix has exponential spectral decay (hence ill-conditioned). We repeat the solve with different keys and compute the norm of the different between approximate solution and the true solution for each key.
        Based on the tolerance level set for the residual, we expect the approximate solutions to be close as well.
        """
        size = 100
        mu = 0.1
        rank = 10
        seed = 0
        num_repeats = 10
        tolerance = 1e-05
        # generate keys
        initial_key = jax.random.PRNGKey(seed)
        generation_key, solver_key = jax.random.split(initial_key)
        # generate PSD matrix with fast spectral decay
        eigvec_key, sol_key = jax.random.split(generation_key)
        Q, _ = jnp.linalg.qr(jax.random.normal(eigvec_key, (size, size)))
        L = jnp.exp(1 / (jnp.arange(size) + 0.5)) - 1
        A = Q @ jnp.diag(L) @ Q.T
        x = jax.random.normal(sol_key, (size,))
        b = A @ x + mu * x
        # run Nyström-PCG
        key = solver_key
        tol_factor = 1 / (L[-1] ** 2 + 2 * L[-1] * mu + mu**2) ** (1 / 2)
        for i in range(num_repeats):
            key, subkey = jax.random.split(key)
            x_final, r_final, status_final, k_final = nystrom_pcg(
                A, b, mu, rank, subkey, tol=tolerance
            )
            # approximate solution should be close to the true solution
            assert jnp.linalg.norm(x - x_final) <= tol_factor * tolerance

    def test_nystrom_pcg_errors(self):
        """
        Test various input errors.
        """
        rank = 1
        mu = 1
        key = jax.random.PRNGKey(0)

        # wrong dimension
        A = 0
        b = 0
        with pytest.raises(
            InputDimError, match="Input A is expected to have dimension 2 but has 0."
        ):
            nystrom_pcg(A, b, mu, rank, key)

        A = jnp.ones(10)
        with pytest.raises(
            InputDimError, match="Input A is expected to have dimension 2 but has 1."
        ):
            nystrom_pcg(A, b, mu, rank, key)

        A = jnp.ones((10, 10, 10))
        with pytest.raises(
            InputDimError, match="Input A is expected to have dimension 2 but has 3."
        ):
            nystrom_pcg(A, b, mu, rank, key)

        A = jnp.ones((10, 10))
        with pytest.raises(
            InputDimError,
            match="Input b is expected to have any dimension in \\[1, 2\\] but has 0.",
        ):
            nystrom_pcg(A, b, mu, rank, key)

        b = jnp.ones((10, 2, 1))
        with pytest.raises(
            InputDimError,
            match="Input b is expected to have any dimension in \\[1, 2\\] but has 3.",
        ):
            nystrom_pcg(A, b, mu, rank, key)

        # wrong shape
        A = A = jnp.ones((10, 5))
        with pytest.raises(
            MatrixNotSquareError,
            match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
        ):
            nystrom_pcg(A, b, mu, rank, key)


class TestSketchySGD:

    def test_sketchysgd_quadratic(self):
        """
        Test SketchySGD on a simple quadratic function.
        With constant Hessian, we use fixed preconditioner and expect the optimizer to reach to the optimum (0) quickly.
        """

        def f(x):
            return jnp.sum(x**2)

        params = jnp.array([1.0, 2.0, 3.0])
        solver = sketchysgd(rank=3, rho=1.0, update_freq=0, seed=0, f=f)
        opt_state = solver.init(params)
        for _ in range(5):
            grad = jax.grad(f)(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

        assert jnp.allclose(f(params), 0.0)

    def test_sketchysgd_quadratic_additional_arg(self):
        """
        Test SketchySGD on a simple quadratic function with additional argument to the objective.
        """

        def f(x, y):
            return jnp.sum(x**2) + jnp.sum(y**2)

        params = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([3.0, 1.0])
        solver = sketchysgd(rank=3, rho=1.0, update_freq=0, seed=0, f=f)
        opt_state = solver.init(params)
        for _ in range(5):
            grad = jax.grad(f)(params, y)
            updates, opt_state = solver.update(grad, opt_state, params, y=y)
            params = optax.apply_updates(params, updates)

        assert jnp.allclose(f(params, y), jnp.sum(y**2))

    def test_sketchysgd_quadratic_pytree(self):
        """
        Test SketchySGD on a simple quadratic function of a tree-like variable.
        """
        Point = namedtuple("Point", ["x", "y"])

        def f(p):
            return (1 / 2) * (p.x**2 + p.y**2)

        params = Point(1.0, 2.0)
        solver = sketchysgd(rank=3, rho=1.0, update_freq=0, seed=0, f=f)
        opt_state = solver.init(params)
        for _ in range(5):
            grad = jax.grad(f)(params)
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

        assert jnp.allclose(f(params), 0.0)
