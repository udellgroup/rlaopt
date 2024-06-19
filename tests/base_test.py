import jax.random
import jax.numpy as jnp
import pytest

from sketchyopts.base import HessianLinearOperator
from collections import namedtuple


class TestHessianLinearOperator:

    def test_pytree(self):
        """
        Test Hessian-vector and Hessian-matrix product computation of the operator.
        For the objectives used in the test, the Hessian should all be identity.
        """
        key = jax.random.PRNGKey(0)

        Point = namedtuple("Point", ["x", "y", "z"])

        params = [
            (1.0, jnp.array([2.0, 3.0])),
            (1.0, {"a": 2.0, "b": 3.0}),
            14**0.5,
            jnp.array([1.0, 2.0, 3.0]),
            Point(1.0, 2.0, 3.0),
        ]

        vectors = [
            jax.random.normal(key, (3, 5)),
            jax.random.normal(key, (3, 2)),
            jax.random.normal(key, (1, 10)),
            jax.random.normal(key, (3, 1)),
            jax.random.normal(key, (3,)),
        ]

        objectives = [
            lambda p: (1 / 2) * (jnp.sum(jnp.square(p[0])) + jnp.sum(jnp.square(p[1]))),
            lambda p: (1 / 2)
            * (
                jnp.sum(jnp.square(p[0]))
                + jnp.sum(jnp.square(p[1]["a"]))
                + jnp.sum(jnp.square(p[1]["b"]))
            ),
            lambda p: (1 / 2) * (p**2),
            lambda p: (1 / 2) * jnp.sum(jnp.square(p)),
            lambda p: (1 / 2)
            * (
                jnp.sum(jnp.square(p.x))
                + jnp.sum(jnp.square(p.y))
                + jnp.sum(jnp.square(p.z))
            ),
        ]

        for p, v, f in zip(params, vectors, objectives):
            assert jnp.allclose(f(p), 7)
            H = HessianLinearOperator(fun=f, grad_fun=None, hvp_fun=None, params=p)
            assert jnp.allclose(H @ v, v)
