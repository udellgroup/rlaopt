# Part of the test utilities are adapted from JAX and JAXopt with modifications.
# - Original JAX type functions:
#   https://github.com/google/jax/blob/main/jax/_src/dtypes.py
# - Original JAXopt test utilities:
#   https://github.com/google/jaxopt/blob/main/jaxopt/_src/test_util.py
#
# Copyright license information:
#
# Copyright 2019 The JAX Authors.
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

import functools
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression

Point = namedtuple("Point", ["x", "y", "z"])


def ridge_regression_sol(X, y, reg):
    """
    Compute the closed-form solution of the ridge regression problem
    (no bias/intercept term).
    """
    n, p = X.shape
    return jnp.linalg.solve(
        (1.0 / n) * X.T @ X + reg * jnp.identity(p), (1.0 / n) * X.T @ y
    )


def ridge_regression_grad(X, y, reg, beta):
    """
    Compute the gradient of the ridge regression objective.
    """
    return (1.0 / X.shape[0]) * X.T @ (X @ beta - y) + reg * beta


def ridge_regression_hessian(X, y, reg, beta):
    """
    Compute the Hessian of the ridge regression objective.
    """
    return (1.0 / X.shape[0]) * X.T @ X + reg * jnp.identity(X.shape[1])


def l2_logistic_regression_sol(X, y, reg):
    """
    Estimate the solution of the l2-regularized logistic regression problem
    (no bias/intercept term) using L-BFGS.
    """
    unscaled_reg = reg * X.shape[0]
    clf = LogisticRegression(
        C=1.0 / unscaled_reg, fit_intercept=False, tol=1e-7, solver="lbfgs"
    )
    clf.fit(X, y)
    return clf.coef_.reshape(-1)


def l2_logistic_regression_grad(X, y, reg, beta):
    """
    Compute the gradient of the l2-regularized logistic regression objective.
    """
    s = 1.0 / (1 + jnp.exp(-y * (X @ beta)))
    return jnp.mean(((s - 1) * y).reshape(-1, 1) * X) + reg * beta


def l2_logistic_regression_hessian(X, y, reg, beta):
    """
    Compute the Hessian of the l2-regularized logistic regression objective.
    """
    s = 1.0 / (1 + jnp.exp(-y * (X @ beta)))
    return (1.0 / X.shape[0]) * X.T @ jnp.diag(s * (1 - s)) @ X + reg * jnp.identity(
        X.shape[1]
    )


def elastic_net_regression_sol(X, y, reg, l1_ratio):
    """
    Estimate the solution of the elastic net-regularized regression problem
    (no bias/intercept term).
    """
    regr = ElasticNet(alpha=reg, l1_ratio=l1_ratio, fit_intercept=False, tol=1e-7)
    regr.fit(X, y)
    return regr.coef_.reshape(-1)


_dtype_to_32bit_dtype = {
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("complex128"): np.dtype("complex64"),
}


@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled, dtype):
    """
    Convert from a dtype to a canonical dtype based on config.x64_enabled.
    """
    try:
        dtype = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f"dtype {dtype!r} not understood") from e

    if x64_enabled:
        return dtype
    else:
        return _dtype_to_32bit_dtype.get(dtype, dtype)


def canonicalize_dtype(dtype):
    return _canonicalize_dtype(jax.config.x64_enabled, dtype)


# default dtypes corresponding to Python scalars
python_scalar_dtypes: dict = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}


def _dtype(x):
    return (
        getattr(x, "dtype", None)
        or np.dtype(python_scalar_dtypes.get(type(x), None))
        or np.asarray(x).dtype
    )


# trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([("float0", np.void, 0)])

_default_tolerance = {
    float0: 0,
    np.dtype(np.bool_): 0,
    np.dtype(np.int8): 0,
    np.dtype(np.int16): 0,
    np.dtype(np.int32): 0,
    np.dtype(np.int64): 0,
    np.dtype(np.uint8): 0,
    np.dtype(np.uint16): 0,
    np.dtype(np.uint32): 0,
    np.dtype(np.uint64): 0,
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-15,
    np.dtype(np.complex64): 1e-6,
    np.dtype(np.complex128): 1e-15,
}


def default_tolerance():
    if device_under_test() != "tpu":
        return _default_tolerance
    tol = _default_tolerance.copy()
    tol[np.dtype(np.float32)] = 1e-3
    tol[np.dtype(np.complex64)] = 1e-3
    return


def tolerance(dtype, tol=None):
    tol = {} if tol is None else tol
    if not isinstance(tol, dict):
        return tol
    tol = {np.dtype(key): value for key, value in tol.items()}
    dtype = canonicalize_dtype(np.dtype(dtype))
    return tol.get(dtype, default_tolerance()[dtype])


def device_under_test():
    return jax.lib.xla_bridge.get_backend().platform


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=""):
    if a.dtype == b.dtype == float0:
        np.testing.assert_array_equal(a, b, err_msg=err_msg)
        return
    kw = {}
    if atol:
        kw["atol"] = atol
    if rtol:
        kw["rtol"] = rtol
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def is_sequence(x):
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


class TestCase:
    """
    Base class for tests.
    """

    def assertArraysEqual(self, x, y, *, check_dtypes=True, err_msg=""):
        """
        Assert that x and y arrays are exactly equal.
        """
        if check_dtypes:
            self.assertDtypesMatch(x, y)
        # work around https://github.com/numpy/numpy/issues/18992
        with np.errstate(over="ignore"):
            np.testing.assert_array_equal(x, y, err_msg=err_msg)

    def assertArraysAllClose(
        self, x, y, *, check_dtypes=True, atol=None, rtol=None, err_msg=""
    ):
        """
        Assert that x and y are close (up to numerical tolerances).
        """
        assert x.shape == y.shape
        atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
        rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

        _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

        if check_dtypes:
            self.assertDtypesMatch(x, y)

    def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
        if not jax.config.x64_enabled and canonicalize_dtypes:
            assert canonicalize_dtype(_dtype(x)) == canonicalize_dtype(_dtype(y))
        else:
            assert _dtype(x) == _dtype(y)

    def assertAllClose(
        self,
        x,
        y,
        *,
        check_dtypes=True,
        atol=None,
        rtol=None,
        canonicalize_dtypes=True,
        err_msg="",
    ):
        """
        Assert that x and y, either arrays or nested tuples/lists, are close.
        """
        if isinstance(x, dict):
            assert isinstance(y, dict), err_msg
            assert set(x.keys()) == set(y.keys()), err_msg
            for k in x.keys():
                self.assertAllClose(
                    x[k],
                    y[k],
                    check_dtypes=check_dtypes,
                    atol=atol,
                    rtol=rtol,
                    canonicalize_dtypes=canonicalize_dtypes,
                    err_msg=err_msg,
                )
        elif is_sequence(x) and not hasattr(x, "__array__"):
            assert is_sequence(y) and not hasattr(y, "__array__"), err_msg
            assert len(x) == len(y), err_msg
            for x_elt, y_elt in zip(x, y):
                self.assertAllClose(
                    x_elt,
                    y_elt,
                    check_dtypes=check_dtypes,
                    atol=atol,
                    rtol=rtol,
                    canonicalize_dtypes=canonicalize_dtypes,
                    err_msg=err_msg,
                )
        elif hasattr(x, "__array__") or np.isscalar(x):
            assert hasattr(y, "__array__") or np.isscalar(
                y
            ), f"{err_msg}: {x} is an array but {y} is not."
            if check_dtypes:
                self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
            x = np.asarray(x)
            y = np.asarray(y)
            self.assertArraysAllClose(
                x, y, check_dtypes=False, atol=atol, rtol=rtol, err_msg=err_msg
            )
        elif x == y:
            return
        else:
            raise TypeError((type(x), type(y)))
