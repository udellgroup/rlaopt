# Tests for proximal operators are adapted from JAXopt with modifications.
# - Original JAXopt proximal operators tests:
#   https://github.com/google/jaxopt/blob/main/tests/prox_test.py
# - Original JAXopt projections tests:
#   https://github.com/google/jaxopt/blob/main/tests/projection_test.py
#
# Copyright license information:
#
# Copyright 2021 Google LLC
# Modifications copyright 2024 the SketchyOpts authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sketchyopts import prox
from sketchyopts.util import ravel_tree, tree_l2_norm, tree_map, tree_ones_like
from tests.test_util import TestCase


class TestProx(TestCase):

    def test_prox_const(self):
        """
        Test prox_const.
        """
        rng = np.random.RandomState(0)
        x = rng.rand(20) * 2 - 1
        self.assertArraysAllClose(prox.prox_const(x), x)

    def _prox_l1(self, x, alpha):
        """
        Scalar implementation for testing.
        """
        if x >= alpha:
            return x - alpha
        elif x <= -alpha:
            return x + alpha
        else:
            return 0

    def test_prox_l1(self):
        """
        Test prox_l1.
        """
        rng = np.random.RandomState(0)

        # check forward pass with array x and scalar alpha.
        x = rng.rand(20) * 2 - 1
        alpha = 0.5
        expected = jnp.array([self._prox_l1(x[i], alpha) for i in range(len(x))])
        got = prox.prox_l1(x, alpha)
        self.assertArraysAllClose(expected, got)

        # check computed Jacobian against manual Jacobian.
        jac = jax.jacobian(prox.prox_l1)(x, alpha)
        jac_exact = np.zeros_like(jac)
        for i in range(len(x)):
            if x[i] >= alpha:
                jac_exact[i, i] = 1
            elif x[i] <= -alpha:
                jac_exact[i, i] = 1
            else:
                jac_exact[i, i] = 0
        self.assertArraysAllClose(jac_exact, jac)

        # check forward pass with array x and array alpha.
        alpha = rng.rand(20)
        expected = jnp.array([self._prox_l1(x[i], alpha[i]) for i in range(len(x))])
        got = prox.prox_l1(x, alpha)
        self.assertArraysAllClose(expected, got)

        # check forward pass with pytree x and pytree alpha.
        x = (rng.rand(20) * 2 - 1, rng.rand(20) * 2 - 1)
        alpha = (rng.rand(20), rng.rand(20))
        expected0 = [self._prox_l1(x[0][i], alpha[0][i]) for i in range(len(x[0]))]
        expected1 = [self._prox_l1(x[1][i], alpha[1][i]) for i in range(len(x[0]))]
        expected = (jnp.array(expected0), jnp.array(expected1))
        got = prox.prox_l1(x, alpha)
        self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

        # check forward pass with pytree x and tuple-of-scalars alpha.
        alpha = (0.5, 0.2)
        expected0 = [self._prox_l1(x[0][i], alpha[0]) for i in range(len(x[0]))]
        expected1 = [self._prox_l1(x[1][i], alpha[1]) for i in range(len(x[0]))]
        expected = (jnp.array(expected0), jnp.array(expected1))
        got = prox.prox_l1(x, alpha)
        self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

        # check forward pass with pytree x and scalar alpha when jit-compiled
        got = jax.jit(prox.prox_l1)(x, 0.5)
        expected0 = [self._prox_l1(x[0][i], 0.5) for i in range(len(x[0]))]
        expected1 = [self._prox_l1(x[1][i], 0.5) for i in range(len(x[0]))]
        expected = (jnp.array(expected0), jnp.array(expected1))
        self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

    def _prox_enet(self, x, lam, gamma):
        """
        Ad-hoc implementation for testing.
        """
        return (1.0 / (1.0 + gamma)) * self._prox_l1(x, lam)

    def test_prox_elastic_net(self):
        """
        Test prox_elastic_net.
        """
        rng = np.random.RandomState(0)

        # check forward pass with array x and scalar hyperparams.
        x = rng.rand(20) * 2 - 1
        hyperparams = (0.5, 0.1)
        expected = jnp.array(
            [self._prox_enet(x[i], *hyperparams) for i in range(len(x))]
        )
        got = prox.prox_elastic_net(x, hyperparams)
        self.assertArraysAllClose(expected, got)

        # check forward pass with array x and array hyperparams.
        hyperparams = (rng.rand(20), rng.rand(20))
        expected = jnp.array(
            [
                self._prox_enet(x[i], hyperparams[0][i], hyperparams[1][i])
                for i in range(len(x))
            ]
        )
        got = prox.prox_elastic_net(x, hyperparams)
        self.assertArraysAllClose(expected, got)

        # check forward pass with pytree x and pytree hyperparams
        x = (rng.rand(20) * 2 - 1, rng.rand(20) * 2 - 1)
        hyperparams = (0.5, 0.1)
        expected0 = [self._prox_enet(x[0][i], *hyperparams) for i in range(len(x[0]))]
        expected1 = [self._prox_enet(x[1][i], *hyperparams) for i in range(len(x[0]))]
        expected = (jnp.array(expected0), jnp.array(expected1))
        got = prox.prox_elastic_net(x, ((0.5, 0.5), (0.1, 0.1)))
        self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

        # check forward pass with pytree x and scalar hyperparams when jit-compiled
        got = jax.jit(prox.prox_elastic_net)(x, hyperparams)
        self.assertArraysAllClose(jnp.array(expected), jnp.array(got))

    def _prox_l2(self, x, alpha):
        """
        Pure NumPy implementation for testing.
        """
        l2_norm = np.sqrt(np.sum(x**2))
        return max(1 - alpha / l2_norm, 0) * x

    def test_prox_l2(self):
        """
        Test prox_l2.
        """
        rng = np.random.RandomState(0)
        x = rng.rand(20) * 2 - 1

        # check non-zero block case
        alpha = 0.1
        got = prox.prox_l2(x, alpha)
        expected = self._prox_l2(x, alpha)
        self.assertArraysAllClose(got, expected)

        # check zero block case
        alpha = 10.0
        got = prox.prox_l2(x, alpha)
        expected = self._prox_l2(x, alpha)
        self.assertArraysAllClose(got, expected)

    def _prox_l2_squared_objective(self, y, alpha, x):
        """
        Proximal operator objective
        f(y) = 0.5 * alpha * ||y||_2^2 + 0.5 * ||y - x||^2
        """
        diff = x - y
        return 0.5 * alpha * jnp.sum(y**2) + 0.5 * jnp.sum(diff**2)

    def test_prox_l2_squared(self):
        """
        Test prox_l2_squared.
        """
        rng = np.random.RandomState(0)
        x = rng.rand(20) * 2 - 1
        x = jnp.array(x)
        alpha = 10.0

        # check the result is indeed the minimizer
        # using the first-order condition (zero gradient)
        got = prox.prox_l2_squared(x, alpha)
        self.assertArraysAllClose(
            jax.grad(self._prox_l2_squared_objective)(got, alpha, x),
            jnp.zeros_like(got),
        )

    def test_prox_nonnegative_l2_squared(self):
        """
        Test prox_nonnegative_l2_squared.
        """
        rng = np.random.RandomState(0)
        x = rng.rand(20) * 2 - 1
        x = jnp.array(x)
        alpha = 10.0

        # check the result is indeed the minimizer
        # using the property of projected gradient
        # i.e. optimal point iff fixed point for convex objective
        got = prox.prox_nonnegative_l2_squared(x, alpha)
        fixed_point = jax.nn.relu(
            got - jax.grad(self._prox_l2_squared_objective)(got, alpha, x)
        )
        self.assertArraysAllClose(got, fixed_point)

    def _prox_l1_objective(self, y, alpha, x):
        """
        Proximal operator objective
        f(y) = alpha * sum(y) + 0.5 ||y - x||^2
        """
        diff = x - y
        return alpha * jnp.sum(y) + 0.5 * jnp.sum(diff**2)

    def test_prox_nonnegative_l1(self):
        rng = np.random.RandomState(0)
        x = rng.rand(20) * 2 - 1
        x = jnp.array(x)
        alpha = 0.5

        # check the result is indeed the minimizer
        # using the property of projected gradient
        # i.e. optimal point iff fixed point for convex objective
        got = prox.prox_nonnegative_l1(x, alpha)
        fixed_point = jax.nn.relu(
            got - jax.grad(self._prox_l1_objective)(got, alpha, x)
        )
        self.assertArraysAllClose(got, fixed_point)

    def test_prox_nonnegative(self):
        """
        Test prox_nonnegative.
        """
        # check forward pass with array x
        x = jnp.array([-1.0, 2.0, 3.0])
        expected = jnp.array([0, 2.0, 3.0])
        self.assertArraysEqual(prox.prox_nonnegative(x), expected)
        self.assertArraysEqual(prox.prox_nonnegative((x, x)), (expected, expected))

        # check forward pass with nested pytree x
        tree_x = (-1.0, {"k1": 1.0, "k2": (1.0, 1.0)}, 1.0)
        tree_expected = (0.0, {"k1": 1.0, "k2": (1.0, 1.0)}, 1.0)
        self.assertAllClose(prox.prox_nonnegative(tree_x), tree_expected)

    def test_prox_box(self):
        """
        Test prox_box.
        """
        # check forward pass with array x
        x = jnp.array([-1.0, 2.0, 3.0])
        expected = jnp.array([0, 2.0, 2.0])
        L, U = 0.0, 2.0
        # lower and upper values are scalars
        self.assertArraysEqual(prox.prox_box(x, (L, U)), expected)
        # lower and upper values are arrays
        L_array = L * jnp.ones(len(x))
        U_array = U * jnp.ones(len(x))
        self.assertArraysEqual(prox.prox_box(x, (L_array, U_array)), expected)

        # check forward pass with pytree x
        self.assertAllClose(
            prox.prox_box((x, x), ((L, L), (U, U))), (expected, expected)
        )
        self.assertAllClose(
            prox.prox_box((x, x), ((L_array, L_array), (U_array, U_array))),
            (expected, expected),
        )

        # check forward pass with nested pytree x
        tree_x = (-1.0, {"k1": 2.0, "k2": (2.0, 3.0)}, 3.0)
        tree_expected = (0.0, {"k1": 2.0, "k2": (2.0, 2.0)}, 2.0)
        # lower and upper values are scalars
        self.assertAllClose(prox.prox_box(tree_x, (L, U)), tree_expected)
        # lower and upper values are pytrees
        U_tree = (2.0, {"k1": 2.0, "k2": (2.0, 2.0)}, 2.0)
        L_tree = (0.0, {"k1": 0.0, "k2": (0.0, 0.0)}, 0.0)
        self.assertAllClose(prox.prox_box(tree_x, (L_tree, U_tree)), tree_expected)

    def test_prox_hyperplane(self):
        """
        Test prox_hyperplane.
        """
        rng = np.random.RandomState(0)

        # check forward pass with array x
        x = rng.randn(50).astype(np.float32)
        a = rng.randn(50).astype(np.float32)
        b = 1.0
        p = prox.prox_hyperplane(x, (a, b))
        self.assertAllClose(jnp.dot(a, p), b)

        # check forward pass with pytree x
        tree_x = (1.0, {"k1": 2.0, "k2": (2.0, 1.0)}, 2.0)
        tree_a = (1.0, {"k1": 1.0, "k2": (1.0, 0.0)}, 1.0)
        p = prox.prox_hyperplane(tree_x, (tree_a, b))
        expected_p = (-0.5, {"k1": 0.5, "k2": (0.5, 1.0)}, 0.5)
        self.assertAllClose(expected_p, p)

    def test_prox_halfspace(self):
        """
        Test prox_halfspace.
        """
        rng = np.random.RandomState(0)
        tree_x = (
            rng.randn(1),
            {"k1": rng.randn(1), "k2": (rng.randn(1), rng.randn(1))},
            rng.randn(1),
        )
        tree_a = (
            rng.randn(1),
            {"k1": rng.randn(1), "k2": (rng.randn(1), rng.randn(1))},
            rng.randn(1),
        )

        # check the case where b is very large (expect x)
        b = 10000
        p = prox.prox_halfspace(tree_x, (tree_a, b))
        self.assertAllClose(tree_x, p)

        # check the case where b is negative (expect projection of x onto hyperplane)
        b = -10000
        p = prox.prox_halfspace(tree_x, (tree_a, b))
        p_hyper = prox.prox_hyperplane(tree_x, (tree_a, b))
        self.assertAllClose(p_hyper, p)

    @pytest.mark.parametrize(
        "prox_op",
        [
            prox.prox_const,
            prox.prox_l1,
            prox.prox_nonnegative_l1,
            prox.prox_elastic_net,
            prox.prox_l2,
            prox.prox_l2_squared,
            prox.prox_nonnegative_l2_squared,
        ],
    )
    def test_pytree_compatibility(self, prox_op):
        """
        Test pytree compatibility.
        """
        rng = np.random.RandomState(0)
        x = dict(a=rng.randn(16, 16), b=rng.randn(16))
        got = prox_op(x)

        # map the prox_op to tree leaves except for prox_l2
        # in which case the prox_op is applied to the unraveled tree
        if prox_op is prox.prox_l2:
            x_raveled, unravel_fun = ravel_tree(x)
            expected = unravel_fun(self._prox_l2(x_raveled, 1.0))
        else:
            expected = tree_map(prox_op, x)

        self.assertAllClose(got, expected)

        # pytree hyperparameters
        if prox_op is prox.prox_l1:
            l1_reg = tree_ones_like(x)
            got = prox_op(x, l1_reg)
            expected = tree_map(prox_op, x, l1_reg)
            self.assertAllClose(got, expected)
        if prox_op is prox.prox_elastic_net:
            hyperparams = [tree_ones_like(x), tree_ones_like(x)]
            got = prox_op(x, hyperparams)
            hyperparams_tree = tree_map(
                lambda y: [jnp.ones_like(y), jnp.ones_like(y)], x
            )
            expected = tree_map(prox_op, x, hyperparams_tree)
            self.assertAllClose(got, expected)
