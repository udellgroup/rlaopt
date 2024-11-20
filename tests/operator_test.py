import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sketchyopts import errors, operator
from tests.test_util import Point, TestCase


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

        data = self.rng.randn(20, 3)
        reg = self.rng.uniform(1)
        vec = self.rng.randn(3).astype("float32")
        mat = self.rng.randn(3, 10).astype("float32")

        prams_size = len(jax._src.flatten_util.ravel_pytree(params)[0])
        expected_shape = (prams_size, prams_size)
        expected_vec = data.T @ data @ vec + reg * vec
        expected_mat = data.T @ data @ mat + reg * mat

        # objective only
        H = operator.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_vec)
        self.assertAllClose(H @ mat, expected_mat)

        # objective + gradient oracle
        H = operator.HessianLinearOperator(
            fun=fun,
            grad_fun=grad_fun,
            hvp_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_vec)
        self.assertAllClose(H @ mat, expected_mat)

        # objective + hvp oracle and square root Hessian oracle
        H = operator.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=hvp_fun,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_vec)
        self.assertAllClose(H @ mat, expected_mat)

    def test_scalar_params(self):
        """
        Test the operator with an objective that accepts a scalar parameter.
        """
        fun = lambda x, data, reg: jnp.log(1 / x) + 0.5 * reg * (x**2)
        grad_fun = lambda x, data, reg: -1 / x + reg * x
        hvp_fun = lambda x, v, data, reg: (1 / jnp.square(x)) * v + reg * v

        params = 10.0
        data = None
        reg = 0.0
        vec = self.rng.randn(1, 10)

        expected_shape = (1, 1)
        expected_result = (1 / (params**2)) * vec + reg * vec

        # objective only
        H = operator.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_result)

        # objective + gradient oracle
        H = operator.HessianLinearOperator(
            fun=fun,
            grad_fun=grad_fun,
            hvp_fun=None,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_result)

        # objective + hvp oracle and square root Hessian oracle
        H = operator.HessianLinearOperator(
            fun=fun,
            grad_fun=None,
            hvp_fun=hvp_fun,
            params=params,
            data=data,
            reg=reg,
        )
        assert H.shape == expected_shape
        self.assertAllClose(H @ vec, expected_result)

    def test_dimension_error(self):
        """
        Test the operator when applied to an array with more than two dimensions.
        """
        params = jnp.ones(3)
        vec = jnp.ones((3, 2, 10))
        H = operator.HessianLinearOperator(
            fun=lambda x, data, reg: jnp.sum(x),
            grad_fun=None,
            hvp_fun=None,
            params=params,
            data=None,
            reg=None,
        )
        with pytest.raises(errors.InputDimError):
            H @ vec
