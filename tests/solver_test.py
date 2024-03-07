import jax.random
import jax.numpy as jnp
import pytest

from sketchyopts.solver import nystrom_pcg
from sketchyopts.errors import InputDimError, MatrixNotSquareError


def test_nystrom_pcg_correctness_1():
    size = 10
    mu = 0
    rank = 1
    seed = 0

    key = jax.random.PRNGKey(seed)
    A = jnp.identity(size)
    b = jnp.ones((size, 1, 1))
    x, _, _ = nystrom_pcg(A, b, mu, rank, key)
    assert jnp.allclose(x, b)


def test_nystrom_pcg_correctness_2():
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
        x_final, r_final, k_final = nystrom_pcg(A, b, mu, rank, subkey, tol=tolerance)
        # approximate solution should be close to the true solution
        assert jnp.linalg.norm(x - x_final) <= tol_factor * tolerance


def test_nystrom_pcg_errors():
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
        InputDimError, match="Input b is expected to have dimension 1 but has 0."
    ):
        nystrom_pcg(A, b, mu, rank, key)

    b = jnp.ones((10, 2))
    with pytest.raises(
        InputDimError, match="Input b is expected to have dimension 1 but has 2."
    ):
        nystrom_pcg(A, b, mu, rank, key)

    # # wrong shape
    A = A = jnp.ones((10, 5))
    with pytest.raises(
        MatrixNotSquareError,
        match="Input A is expected to be a square matrix but has shape \\(10, 5\\).",
    ):
        nystrom_pcg(A, b, mu, rank, key)
