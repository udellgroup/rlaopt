import jax.random
import jax.numpy as jnp
import pytest

from sketchyopts.preconditioner import rand_nystrom_approx
from sketchyopts.errors import InputDimError, MatrixNotSquareError


def test_rand_nystrom_approx_correctness():
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


def test_rand_nystrom_approx_errors():
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

    # # wrong shape
    A = A = jnp.ones((10, 5))
    with pytest.raises(
        MatrixNotSquareError,
        match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
    ):
        rand_nystrom_approx(A, rank, key)
