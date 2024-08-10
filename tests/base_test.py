from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sketchyopts import base, errors
from tests.test_util import (
    Point,
    TestCase,
    ridge_regression_grad,
    ridge_regression_hessian,
)


class TestHessianLinearOperator(TestCase):

    rng = np.random.RandomState(0)

    @pytest.mark.parametrize(
        "params",
        [
            rng.randn(3),
            (rng.randn(1), rng.randn(2)),
            (rng.randn(1), {"a": rng.randn(1), "b": rng.randn(1)}),
            Point(rng.randn(1), rng.randn(1), rng.randn(1)),
        ],
    )
    def test_ridge_obective(self, params):
        """
        Test the operator with the objective of ridge regression:

            0.5 * || data @ params ||_2^2 + 0.5 * reg * || params ||_2^2

        The operator should work for any pytree structure for ``params``.
        """

        def fun(p, data, reg):
            raveled_p, _ = jax._src.flatten_util.ravel_pytree(p)
            return 0.5 * jnp.sum(jnp.square(data @ raveled_p)) + 0.5 * reg * jnp.sum(
                jnp.square(raveled_p)
            )

        def grad_fun(p, data, reg):
            raveled_p, ravel_fun = jax._src.flatten_util.ravel_pytree(p)
            return ravel_fun(data.T @ data @ raveled_p + reg * raveled_p)

        def hvp_fun(p, v, data, reg):
            _, ravel_fun = jax._src.flatten_util.ravel_pytree(p)
            raveled_v, _ = jax._src.flatten_util.ravel_pytree(v)
            return ravel_fun(data.T @ data @ raveled_v + reg * raveled_v)

        def sqrt_hess_fun(p, data):
            return data

        data = self.rng.randn(20, 3)
        reg = self.rng.uniform(1)
        vec = self.rng.randn(3).astype("float32")
        mat = self.rng.randn(3, 10).astype("float32")

        prams_size = len(jax._src.flatten_util.ravel_pytree(params)[0])
        expected_shape = (prams_size, prams_size)
        expected_vec = data.T @ data @ vec + reg * vec
        expected_mat = data.T @ data @ mat + reg * mat
        expected_H = data.T @ data + reg * jnp.identity(prams_size)

        # objective only
        H = base.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=None,
            sqrt_hess_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_vec)
        self.assertAllClose(H @ mat, expected_mat)
        self.assertAllClose(H.as_matrix(), expected_H)

        # objective + gradient oracle
        H = base.HessianLinearOperator(
            fun=fun,
            grad_fun=grad_fun,
            hvp_fun=None,
            sqrt_hess_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        print(H.as_matrix())
        self.assertAllClose(H @ vec, expected_vec)
        self.assertAllClose(H @ mat, expected_mat)
        self.assertAllClose(H.as_matrix(), expected_H)

        # objective + hvp oracle and square root Hessian oracle
        H = base.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=hvp_fun,
            sqrt_hess_fun=sqrt_hess_fun,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_vec)
        self.assertAllClose(H @ mat, expected_mat)
        self.assertAllClose(H.as_matrix(), expected_H)

    def test_scalar_params(self):
        """
        Test the operator with an objective that accepts a scalar parameter.
        """
        fun = lambda x, data, reg: jnp.log(1 / x) + 0.5 * reg * (x**2)
        grad_fun = lambda x, data, reg: -1 / x + reg * x
        hvp_fun = lambda x, v, data, reg: (1 / jnp.square(x)) * v + reg * v
        sqrt_hess_fun = lambda x, data: (1 / x) * jnp.ones((1, 1))

        params = 10.0
        data = None
        reg = 0.0
        vec = self.rng.randn(1, 10)

        expected_shape = (1, 1)
        expected_result = (1 / (params**2)) * vec + reg * vec
        expected_H = (1 / (params**2)) * jnp.ones((1, 1))

        # objective only
        H = base.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=None,
            sqrt_hess_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_result)
        self.assertAllClose(H.as_matrix(), expected_H)

        # objective + gradient oracle
        H = base.HessianLinearOperator(
            fun=fun,
            grad_fun=grad_fun,
            hvp_fun=None,
            sqrt_hess_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_result)
        self.assertAllClose(H.as_matrix(), expected_H)

        # objective + hvp oracle and square root Hessian oracle
        H = base.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=hvp_fun,
            sqrt_hess_fun=sqrt_hess_fun,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_result)
        self.assertAllClose(H.as_matrix(), expected_H)

    def test_dimension_error(self):
        """
        Test the operator when applied to an array with more than two dimensions.
        """
        params = jnp.ones(3)
        vec = jnp.ones((3, 2, 10))
        H = base.HessianLinearOperator(
            fun=lambda x, data, reg: jnp.sum(x),
            grad_fun=None,
            hvp_fun=None,
            params=params,
            data=None,
            reg=None,
        )
        with pytest.raises(errors.InputDimError):
            H @ vec


SolverInteralState = namedtuple(
    "SolverInteralState",
    ["step_size", "iter_num", "key", "precond"],
    defaults=[0, jax.random.PRNGKey(0), None],
)


def obj_fun(beta, data, reg):
    """
    Objective function of ridge regression.
    """
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


def grad_fun(beta, data, reg):
    """
    Gradient function of ridge regression.
    """
    features = data[:, :-1]
    targets = data[:, -1]
    unraveled_beta, unravel_fun = jax._src.flatten_util.ravel_pytree(beta)
    return unravel_fun(ridge_regression_grad(features, targets, reg, unraveled_beta))


def sqrt_hess_fun(beta, data):
    """
    Square root Hessian function of ridge regression.
    """
    sqrt_n = data.shape[0] ** 0.5
    features = data[:, :-1]
    return (1 / sqrt_n) * features


class TestPromiseSolverClass(TestCase):

    key = jax.random.PRNGKey(0)

    # generate data
    num_samples = 100
    num_features = 10
    reg = 0.1
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    X = jax.random.normal(subkey1, (num_samples, num_features))
    y = jax.random.normal(subkey2, (num_samples,))
    data = jnp.hstack([X, jnp.expand_dims(y, 1)])

    # initial parameters
    beta_0 = (jnp.zeros(2), jnp.zeros(3), jnp.zeros(5))

    # solver hyperparameters
    rho = 0.1
    rank = 10
    grad_batch_size = 10
    hess_batch_size = 25
    update_freq = 0
    seed = 0

    # initialize solver objects
    solver_nyssn = base.PromiseSolver(
        fun=obj_fun,
        rho=rho,
        rank=rank,
        grad_batch_size=grad_batch_size,
        hess_batch_size=hess_batch_size,
        update_freq=update_freq,
        seed=seed,
    )

    solver_ssn = base.PromiseSolver(
        fun=obj_fun,
        sqrt_hess_fun=sqrt_hess_fun,
        precond="ssn",
        rho=rho,
        grad_batch_size=grad_batch_size,
        hess_batch_size=hess_batch_size,
        update_freq=update_freq,
        seed=seed,
    )

    # manually set class variables (since these are set to the actual values only inside the run function)
    solver_nyssn.num_samples = num_samples
    solver_ssn.num_samples = num_samples

    solver_nyssn.params_len = num_features
    solver_ssn.params_len = num_features

    def test_value_and_grad_fun(self):
        """
        Test the ``_value_and_grad`` function of the solver.
        The function should return correct objective and gradient values.
        """
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, num=4)

        # generate random parameters
        params = (
            jax.random.normal(subkey1, (2,)),
            jax.random.normal(subkey2, (3,)),
            jax.random.normal(subkey3, (5,)),
        )

        # with only the objective function
        # for the Nyström subsampled Newton preconditioner
        val, grad = self.solver_nyssn._value_and_grad(
            params, data=self.data, reg=self.reg
        )  # params should be the first positional argument
        self.assertAllClose(val, obj_fun(beta=params, data=self.data, reg=self.reg))
        self.assertAllClose(grad, grad_fun(beta=params, data=self.data, reg=self.reg))
        # for the subsampled Newton preconditioner
        val, grad = self.solver_ssn._value_and_grad(
            params, data=self.data, reg=self.reg
        )  # params should be the first positional argument
        self.assertAllClose(val, obj_fun(beta=params, data=self.data, reg=self.reg))
        self.assertAllClose(grad, grad_fun(beta=params, data=self.data, reg=self.reg))

        # with both the objective and gradient functions
        # for the Nyström subsampled Newton preconditioner
        solver = base.PromiseSolver(
            fun=obj_fun,
            grad_fun=grad_fun,
            rho=self.rho,
            rank=self.rank,
            grad_batch_size=self.grad_batch_size,
            hess_batch_size=self.hess_batch_size,
            update_freq=self.update_freq,
            seed=self.seed,
        )
        val, grad = solver._value_and_grad(
            params, data=self.data, reg=self.reg
        )  # params should be the first positional argument
        self.assertAllClose(val, obj_fun(beta=params, data=self.data, reg=self.reg))
        self.assertAllClose(grad, grad_fun(beta=params, data=self.data, reg=self.reg))
        # for the subsampled Newton preconditioner
        solver = base.PromiseSolver(
            fun=obj_fun,
            grad_fun=grad_fun,
            sqrt_hess_fun=sqrt_hess_fun,
            precond="ssn",
            rho=self.rho,
            grad_batch_size=self.grad_batch_size,
            hess_batch_size=self.hess_batch_size,
            update_freq=self.update_freq,
            seed=self.seed,
        )
        val, grad = solver._value_and_grad(
            params, data=self.data, reg=self.reg
        )  # params should be the first positional argument
        self.assertAllClose(val, obj_fun(beta=params, data=self.data, reg=self.reg))
        self.assertAllClose(grad, grad_fun(beta=params, data=self.data, reg=self.reg))

    def test_grad_transform_fun_nyssn(self):
        """
        Test the ``_get_grad_transform`` function of the solve with the Nyström subsampled Newton preconditioner.
        The function should return an oracle that computes :math:`P^{-1} g` where :math:`P` is the preconditioner and :math:`g` is any vector of matching size.
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)

        # generate Hessian
        H_sqrt = jax.random.normal(subkey1, (self.num_features, self.num_features))
        H = H_sqrt.T @ H_sqrt  # generate psd matrix

        # compute decomposition of the Hessian
        U, S, _ = jnp.linalg.svd(H)

        # generate random vector
        v = jax.random.normal(subkey2, (self.num_features,))

        # manually compute preconditioned vector
        precond = H + self.rho * jnp.identity(self.num_features)
        precond_v = jnp.linalg.solve(precond, v)

        self.assertAllClose(
            precond_v, self.solver_nyssn._grad_transform(v, (U, S)), atol=1e-04
        )

    def test_grad_transform_fun_ssn(self):
        """
        Test the ``_get_grad_transform`` function of the solve with the subsampled Newton preconditioner.
        The function should return an oracle that computes :math:`P^{-1} g` where :math:`P` is the preconditioner and :math:`g` is any vector of matching size.
        """
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, num=4)

        wide_dim = int(self.num_features / 2)
        tall_dim = int(self.num_features * 2)

        # generate Hessian
        H_sqrt_wide = jax.random.normal(subkey1, (wide_dim, self.num_features))
        H_sqrt_tall = jax.random.normal(subkey2, (tall_dim, self.num_features))
        H_wide = H_sqrt_wide.T @ H_sqrt_wide  # generate psd matrix
        H_tall = H_sqrt_tall.T @ H_sqrt_tall  # generate psd matrix

        # compute decomposition of the Hessian
        L_wide = jnp.linalg.cholesky(
            H_sqrt_wide @ H_sqrt_wide.T + self.rho * jnp.identity(wide_dim), upper=False
        )
        L_tall = jnp.linalg.cholesky(
            H_sqrt_tall.T @ H_sqrt_tall + self.rho * jnp.identity(self.num_features),
            upper=False,
        )

        # generate random vector
        v = jax.random.normal(subkey3, (self.num_features,))

        # manually compute preconditioned vector
        precond_wide = H_wide + self.rho * jnp.identity(self.num_features)
        precond_tall = H_tall + self.rho * jnp.identity(self.num_features)
        precond_v_wide = jnp.linalg.solve(precond_wide, v)
        precond_v_tall = jnp.linalg.solve(precond_tall, v)

        self.assertAllClose(
            precond_v_wide,
            self.solver_ssn._grad_transform(v, (L_wide, H_sqrt_wide)),
            atol=1e-04,
        )
        self.assertAllClose(
            precond_v_tall,
            self.solver_ssn._grad_transform(v, (L_tall, H_sqrt_tall)),
            atol=1e-04,
        )

    def test_precond_smoothness_constant_estimate(self):
        """
        Test the ``_estimate_constant`` function of the solver.
        The function should return the largest eigenvalue of :math:`P^{-1/2} H P^{-1/2}` where :math:`P` is the preconditioner and :math:`H` is the Hessian.
        """
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, num=4)

        # generate Hessian for the objective sans the regularization
        H_sqrt = jax.random.normal(subkey1, (self.num_features, self.num_features))
        H = H_sqrt.T @ H_sqrt  # generate psd matrix

        # compute decomposition of the Hessian
        U, S, _ = jnp.linalg.svd(H)
        L = jnp.linalg.cholesky(
            H + self.rho * jnp.identity(self.num_features), upper=False
        )

        # obtained the constant estimate
        H_operator = base.HessianLinearOperator(
            fun=obj_fun,
            grad_fun=None,
            hvp_fun=None,
            sqrt_hess_fun=sqrt_hess_fun,
            params=self.beta_0,
            data=self.data,
            reg=self.reg,
        )

        returned_labda_nyssn = self.solver_nyssn._estimate_constant(
            H_operator, (U, S), subkey2
        )
        returned_labda_ssn = self.solver_ssn._estimate_constant(
            H_operator, (L, H_sqrt), subkey3
        )

        # compute the constant manually
        H_true = ridge_regression_hessian(
            self.data[:, :-1], self.data[:, -1], self.reg, self.beta_0
        )
        inv_sqrt_precond = U @ jnp.diag(1.0 / jnp.sqrt(S + self.rho)) @ U.T
        expected_labda = jnp.linalg.eigvals(
            inv_sqrt_precond @ H_true @ inv_sqrt_precond
        )[0].real

        self.assertAllClose(expected_labda, returned_labda_nyssn, atol=1e-04)
        self.assertAllClose(expected_labda, returned_labda_ssn, atol=1e-04)

    def test_step_size_update(self):
        """
        Test the ``_update_step_size`` function of the solver.
        The function should update the ``step_size`` attribute with :math:`\alpha / \\lambda_P` where :math:`\alpha` is the learning rate multiplier and :math:`\\lambda_P` is the smoothness constant of the preconditioner.
        """
        lr = 0.5
        labda = 10
        # specify the learning rate multiplier for the solver
        self.solver_nyssn.learning_rate = lr
        self.solver_ssn.learning_rate = lr
        # substantiate state object
        state = SolverInteralState(step_size=0.0)
        # update the step size using the built-in class method
        returned_state_nyssn = self.solver_nyssn._update_step_size(labda, state)
        returned_state_ssn = self.solver_ssn._update_step_size(labda, state)
        # manually construct the expected updated state
        expected_state = SolverInteralState(step_size=lr / labda)

        assert expected_state == returned_state_nyssn
        assert expected_state == returned_state_ssn

    def test_precond_update(self):
        """
        Test the ``_update_precond`` function of the solver.
        The function should update the ``step_size`` and ``precond`` attributes.
        """
        self.key, subkey = jax.random.split(self.key)

        # set the Hessian batch size to the total number of samples in the dataset
        self.solver_nyssn.hess_batch_size = self.num_samples
        self.solver_ssn.hess_batch_size = self.num_samples

        # generate solver state
        iter_num = 10
        state = SolverInteralState(step_size=0.0, iter_num=iter_num)

        # compute true Hessian
        H = ridge_regression_hessian(
            self.data[:, :-1], self.data[:, -1], self.reg, self.beta_0
        )

        # compute smoothness constant of the preconditioned Hessian
        labda = 1.0

        # generate random vector
        v = jax.random.normal(subkey, (self.num_features,))

        # manually compute preconditioned vector
        # precond = H + self.rho * jnp.identity(self.num_features)
        precond_v = jnp.linalg.solve(H, v)

        # test the case when the learning rate is passed in as a schedule
        self.solver_nyssn.learning_rate = lambda x: 1 / x
        self.solver_ssn.learning_rate = lambda x: 1 / x
        # for the Nyström subsampled Newton preconditioner
        returned_state = self.solver_nyssn._update_precond(
            self.beta_0, state, self.data, reg=self.reg
        )
        assert returned_state.step_size == self.solver_nyssn.learning_rate(iter_num)
        self.assertAllClose(
            self.solver_nyssn._grad_transform(v, returned_state.precond),
            precond_v,
            atol=1e-05,
        )
        # for the subsampled Newton preconditioner
        returned_state = self.solver_ssn._update_precond(
            self.beta_0, state, self.data, reg=self.reg
        )
        assert returned_state.step_size == self.solver_ssn.learning_rate(iter_num)
        H_t = sqrt_hess_fun(self.beta_0, self.data).T @ sqrt_hess_fun(
            self.beta_0, self.data
        ) + self.rho * jnp.identity(self.num_features)
        print(jnp.linalg.solve(H_t, v))
        self.assertAllClose(
            self.solver_ssn._grad_transform(v, returned_state.precond),
            precond_v,
            atol=1e-05,
        )

        # test the case when the learning rate is passed in as a multiplier
        lr = 0.5
        self.solver_nyssn.learning_rate = lr
        self.solver_ssn.learning_rate = lr
        # we override the smoothness constant estimation function with the one that runs for more iterations
        # for the Nyström subsampled Newton preconditioner
        est_const = self.solver_nyssn._estimate_constant
        self.solver_nyssn._estimate_constant = (
            lambda H_S, grad_transform, key: est_const(
                H_S, grad_transform, key, p_maxiter=500
            )
        )
        returned_state = self.solver_nyssn._update_precond(
            self.beta_0, state, self.data, reg=self.reg
        )
        self.assertAllClose(returned_state.step_size, lr / labda)
        self.assertAllClose(
            self.solver_nyssn._grad_transform(v, returned_state.precond),
            precond_v,
            atol=1e-05,
        )
        # for the subsampled Newton preconditioner
        est_const = self.solver_ssn._estimate_constant
        self.solver_ssn._estimate_constant = lambda H_S, grad_transform, key: est_const(
            H_S, grad_transform, key, p_maxiter=500
        )
        returned_state = self.solver_ssn._update_precond(
            self.beta_0, state, self.data, reg=self.reg
        )
        self.assertAllClose(returned_state.step_size, lr / labda)
        self.assertAllClose(
            self.solver_ssn._grad_transform(v, returned_state.precond),
            precond_v,
            atol=1e-05,
        )
