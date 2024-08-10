import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sketchyopts import errors, solver
from tests.test_util import TestCase, l2_logistic_regression_sol, ridge_regression_sol


class TestNystromPCG(TestCase):

    key = jax.random.PRNGKey(0)

    def test_nystrom_pcg_correctness_1(self):
        """
        Test a trivial linear system with a single righthand side.
        The matrix is identity and we expect the solution to match the generated righthand side.
        """
        size = 10
        mu = 0
        rank = 1

        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        A = jnp.identity(size)
        b = jax.random.normal(subkey1, (size, 1))
        x, _, status, k = solver.nystrom_pcg(A, b, mu, rank, subkey2)

        self.assertAllClose(x, b)
        self.assertAllClose(status, jnp.ones(1, dtype=bool))

    def test_nystrom_pcg_multiple_righthand_sides(self):
        """
        Test a trivial linear system with multiple righthand sides.
        The matrix is identity and we expect the solution to match the generated righthand sides.
        """
        size = 10
        num_rhs = 20
        mu = 0
        rank = 1

        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        A = jnp.identity(size)
        b = jax.random.normal(subkey1, (size, num_rhs))
        x, _, status, k = solver.nystrom_pcg(A, b, mu, rank, subkey2)

        self.assertAllClose(x, b)
        self.assertAllClose(status, jnp.ones(num_rhs, dtype=bool))
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
        num_repeats = 10
        tolerance = 1e-05

        # generate PSD matrix with fast spectral decay
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        Q, _ = jnp.linalg.qr(jax.random.normal(subkey1, (size, size)))
        L = jnp.exp(1 / (jnp.arange(size) + 0.5)) - 1
        A = Q @ jnp.diag(L) @ Q.T
        x = jax.random.normal(subkey2, (size,))
        b = A @ x + mu * x

        # run Nyström-PCG
        tol_factor = 1 / (L[-1] ** 2 + 2 * L[-1] * mu + mu**2) ** (1 / 2)
        for i in range(num_repeats):
            self.key, subkey = jax.random.split(self.key)
            x_final, r_final, status_final, k_final = solver.nystrom_pcg(
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

        # wrong dimension
        A = 0
        b = 0
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input A is expected to have dimension 2 but has 0.",
        ):
            solver.nystrom_pcg(A, b, mu, rank, subkey)

        A = jnp.ones(10)
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input A is expected to have dimension 2 but has 1.",
        ):
            solver.nystrom_pcg(A, b, mu, rank, subkey)

        A = jnp.ones((10, 10, 10))
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input A is expected to have dimension 2 but has 3.",
        ):
            solver.nystrom_pcg(A, b, mu, rank, subkey)

        A = jnp.ones((10, 10))
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input b is expected to have any dimension in \\[1, 2\\] but has 0.",
        ):
            solver.nystrom_pcg(A, b, mu, rank, subkey)

        b = jnp.ones((10, 2, 1))
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.InputDimError,
            match="Input b is expected to have any dimension in \\[1, 2\\] but has 3.",
        ):
            solver.nystrom_pcg(A, b, mu, rank, subkey)

        # wrong shape
        A = jnp.ones((10, 5))
        self.key, subkey = jax.random.split(self.key)
        with pytest.raises(
            errors.MatrixNotSquareError,
            match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
        ):
            solver.nystrom_pcg(A, b, mu, rank, subkey)


def ridge_obj_fun(beta, data, reg):
    """
    Objective function of ridge regression.
    The function works for both one-dimensional (single sample) or two-dimensional (a batch of samples) data input.
    """
    if jnp.ndim(data) == 1:
        features = jnp.expand_dims(data[:-1], axis=0)
        targets = jnp.expand_dims(data[-1], axis=0)
    else:
        features = data[:, :-1]
        targets = data[:, -1]
    preds = (
        features[:, :2] @ beta[0]
        + features[:, 2:5] @ beta[1]
        + features[:, 5:] @ beta[2]
    )
    res = targets - preds
    return (1 / 2) * jnp.mean(jnp.square(res)) + (reg / 2) * (
        jnp.sum(jnp.square(beta[0]))
        + jnp.sum(jnp.square(beta[1]))
        + jnp.sum(jnp.square(beta[2]))
    )


def ridge_sqrt_hess_fun(beta, data):
    """
    Square root Hessian function of ridge regression.
    """
    sqrt_n = data.shape[0] ** 0.5
    features = data[:, :-1]
    return (1 / sqrt_n) * features


def logistic_obj_fun(beta, data, reg):
    """
    Objective function of logistic regression.
    The function works for both one-dimensional (single sample) or two-dimensional (a batch of samples) data input.
    """
    if jnp.ndim(data) == 1:
        features = jnp.expand_dims(data[:-1], axis=0)
        labels = jnp.expand_dims(data[-1], axis=0)
    else:
        features = data[:, :-1]
        labels = data[:, -1]
    log_terms = jnp.log(
        1
        + jnp.exp(
            -labels
            * (
                features[:, :2] @ beta[0]
                + features[:, 2:5] @ beta[1]
                + features[:, 5:] @ beta[2]
            )
        )
    )
    return jnp.mean(log_terms) + (reg / 2) * (
        jnp.sum(jnp.square(beta[0]))
        + jnp.sum(jnp.square(beta[1]))
        + jnp.sum(jnp.square(beta[2]))
    )


def logistic_sqrt_hess_fun(beta, data):
    """
    Square root Hessian function of logistic regression.
    """
    sqrt_n = data.shape[0] ** 0.5
    features = data[:, :-1]
    scores = 1.0 / (
        1
        + jnp.exp(
            features[:, :2] @ beta[0]
            + features[:, 2:5] @ beta[1]
            + features[:, 5:] @ beta[2]
        )
    )
    return (1 / sqrt_n) * jnp.sqrt((scores * (1 - scores))).reshape(-1, 1) * features


class TestPromiseSolvers(TestCase):

    key = jax.random.PRNGKey(0)

    # generate data
    num_samples = 20
    num_features = 10
    reg = 0.01
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    X = jax.random.normal(subkey1, (num_samples, num_features))

    y_ridge = jax.random.normal(subkey2, (num_samples,))
    data_ridge = jnp.hstack([X, jnp.expand_dims(y_ridge, 1)])

    y_logistic = 2 * jax.random.randint(subkey2, (num_samples,), 0, 2) - 1
    data_logistic = jnp.hstack([X, jnp.expand_dims(y_logistic, 1)])

    # initial parameters
    beta_0 = (jnp.zeros(2), jnp.zeros(3), jnp.zeros(5))

    # compute reference solution
    _, unravel_fun = jax._src.flatten_util.ravel_pytree(beta_0)

    ridge_beta_sol = unravel_fun(ridge_regression_sol(X, y_ridge, reg))
    ridge_value_sol = ridge_obj_fun(ridge_beta_sol, data_ridge, reg)

    logistic_beta_sol = unravel_fun(l2_logistic_regression_sol(X, y_logistic, reg))
    logistic_value_sol = logistic_obj_fun(logistic_beta_sol, data_logistic, reg)

    # solver hyperparameters
    rho = 0.1
    rank = 10
    grad_batch_size = 20
    hess_batch_size = 20
    update_freq_ridge = 0
    update_freq_logistic = 10
    seed = 0
    maxiter = 100
    tol = 1e-05

    @pytest.mark.parametrize(
        "promise_solver",
        [
            (solver.SketchySGD, {"learning_rate": 0.5}),
            (solver.SketchySVRG, {"learning_rate": 0.5, "snapshop_update_freq": 1}),
            (solver.SketchySAGA, {"learning_rate": 0.5}),
            (
                solver.SketchyKatyusha,
                {
                    "mu": 0.1,
                    "snapshop_update_prob": 0.5,
                },
            ),
        ],
    )
    def test_ridge_regression(self, promise_solver):
        """
        Test PROMISE solvers on a ridge regression problem.
        The parameters we seek to optimize has a tree-like structure.
        The problem has a constant Hessian and therefore we do not update the preconditioner during the run.
        """
        solver_class, solver_params = promise_solver

        # solver with the Nyström subsampled Newton preconditioner
        solver = solver_class(
            fun=ridge_obj_fun,
            rank=self.rank,
            rho=self.rho,
            grad_batch_size=self.grad_batch_size,
            hess_batch_size=self.hess_batch_size,
            update_freq=self.update_freq_ridge,
            seed=self.seed,
            maxiter=self.maxiter,
            tol=self.tol,
            **solver_params,
        )
        beta_final, solver_state = solver.run(
            self.beta_0, self.data_ridge, reg=self.reg
        )

        self.assertAllClose(
            self.ridge_value_sol, ridge_obj_fun(beta_final, self.data_ridge, self.reg)
        )
        assert solver_state.iter_num <= self.maxiter
        assert solver_state.error <= self.tol

        # solver with the subsampled Newton preconditioner
        solver = solver_class(
            fun=ridge_obj_fun,
            sqrt_hess_fun=ridge_sqrt_hess_fun,
            precond="ssn",
            rho=self.rho,
            grad_batch_size=self.grad_batch_size,
            hess_batch_size=self.hess_batch_size,
            update_freq=self.update_freq_ridge,
            seed=self.seed,
            maxiter=self.maxiter,
            tol=self.tol,
            **solver_params,
        )
        beta_final, solver_state = solver.run(
            self.beta_0, self.data_ridge, reg=self.reg
        )

        self.assertAllClose(
            self.ridge_value_sol, ridge_obj_fun(beta_final, self.data_ridge, self.reg)
        )
        assert solver_state.iter_num <= self.maxiter
        assert solver_state.error <= self.tol

    @pytest.mark.parametrize(
        "promise_solver",
        [
            (solver.SketchySGD, {"learning_rate": 0.5}),
            (solver.SketchySVRG, {"learning_rate": 0.5, "snapshop_update_freq": 3}),
            (solver.SketchySAGA, {"learning_rate": 0.5}),
            (
                solver.SketchyKatyusha,
                {
                    "mu": 0.1,
                    "snapshop_update_prob": 0.5,
                },
            ),
        ],
    )
    def test_logistic_regression(self, promise_solver):
        """
        Test PROMISE solvers on a logistic regression problem.
        The parameters we seek to optimize has a tree-like structure.
        We update the preconditioner at each iteration during the run.
        """
        solver_class, solver_params = promise_solver

        # solver with the Nyström subsampled Newton preconditioner
        solver = solver_class(
            fun=logistic_obj_fun,
            rank=self.rank,
            rho=self.rho,
            grad_batch_size=self.grad_batch_size,
            hess_batch_size=self.hess_batch_size,
            update_freq=self.update_freq_logistic,
            seed=self.seed,
            maxiter=self.maxiter,
            tol=self.tol,
            jit=True,
            **solver_params,
        )
        beta_final, solver_state = solver.run(
            self.beta_0, self.data_logistic, reg=self.reg
        )

        self.assertAllClose(
            self.logistic_value_sol,
            logistic_obj_fun(beta_final, self.data_logistic, self.reg),
        )
        assert solver_state.iter_num <= self.maxiter
        assert solver_state.error <= self.tol

        # solver with the subsampled Newton preconditioner
        solver = solver_class(
            fun=logistic_obj_fun,
            sqrt_hess_fun=logistic_sqrt_hess_fun,
            precond="ssn",
            rho=self.rho,
            grad_batch_size=self.grad_batch_size,
            hess_batch_size=self.hess_batch_size,
            update_freq=self.update_freq_logistic,
            seed=self.seed,
            maxiter=self.maxiter,
            tol=self.tol,
            **solver_params,
        )
        beta_final, solver_state = solver.run(
            self.beta_0, self.data_logistic, reg=self.reg
        )

        self.assertAllClose(
            self.logistic_value_sol,
            logistic_obj_fun(beta_final, self.data_logistic, self.reg),
        )
        assert solver_state.iter_num <= self.maxiter
        assert solver_state.error <= self.tol
