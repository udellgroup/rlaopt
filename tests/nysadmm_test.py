import jax
import jax.numpy as jnp
import numpy as np

from sketchyopts import prox, solver
from tests.test_util import TestCase, elastic_net_regression_sol


def least_squares_obj_fun(beta, data):
    """
    Objective function of least squares.
    """
    features = data[:, :-1]
    targets = data[:, -1]
    preds = (
        features[:, :2] @ beta[0]
        + features[:, 2:5] @ beta[1]
        + features[:, 5:] @ beta[2]
    )
    res = targets - preds
    return (1 / 2) * jnp.mean(jnp.square(res))


def l2_squared_reg(beta, scaling):
    """
    Function representing the l2 squared regularization.
    """
    return (0.5 * scaling) * (
        jnp.sum(jnp.square(beta[0]))
        + jnp.sum(jnp.square(beta[1]))
        + jnp.sum(jnp.square(beta[2]))
    )


class TestNysADMM(TestCase):

    key = jax.random.PRNGKey(0)

    # generate data
    num_samples = 20
    num_features = 10
    reg = 0.01
    l1_ratio = 0.2
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    X = jax.random.normal(subkey1, (num_samples, num_features))

    y = jax.random.normal(subkey2, (num_samples,))
    data = jnp.hstack([X, jnp.expand_dims(y, 1)])

    # initial parameters
    beta_0 = (jnp.ones(2), jnp.ones(3), jnp.ones(5))

    # compute reference solution
    _, unravel_fun = jax._src.flatten_util.ravel_pytree(beta_0)

    beta_sol = unravel_fun(elastic_net_regression_sol(X, y, reg, l1_ratio))

    # solver hyperparameters
    step_size = 0.1
    sketch_size = 10
    update_freq = 0
    maxiter: int = 20
    abs_tol = 1e-7
    rel_tol = 1e-7
    fun_params = {}
    reg_g_params = {"scaling": reg * (1 - l1_ratio)}
    prox_reg_h_params = {"l1reg": reg * l1_ratio}

    def test_elastic_net_regression(self):
        """
        Test NysADMM on an elastic net regression problem.

        The parameters we seek to optimize has a tree-like structure.
        """
        opt = solver.NysADMM(
            fun=least_squares_obj_fun,
            reg_g=l2_squared_reg,
            prox_reg_h=prox.prox_l1,
            step_size=self.step_size,
            sketch_size=self.sketch_size,
            update_freq=self.update_freq,
            maxiter=self.maxiter,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
        )
        beta_opt, opt_state = opt.run(
            self.beta_0,
            self.data,
            self.fun_params,
            self.reg_g_params,
            self.prox_reg_h_params,
        )
        self.assertAllClose(beta_opt, self.beta_sol)
        assert opt_state.iter_num <= self.maxiter
        primal_residual_norm = np.linalg.norm(
            jax._src.flatten_util.ravel_pytree(opt_state.res_primal)[0]
        )
        dual_residual_norm = np.linalg.norm(
            jax._src.flatten_util.ravel_pytree(opt_state.res_dual)[0]
        )
        assert primal_residual_norm <= self.abs_tol
        assert dual_residual_norm <= self.abs_tol
