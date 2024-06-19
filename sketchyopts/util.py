import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from jax import Array

KeyArray = Array
KeyArrayLike = ArrayLike


def generate_random_batch(
    data: Array, batch_size: int, key: KeyArray
) -> tuple[Array, KeyArray]:
    n = jnp.shape(data)[0]
    key, subkey = jax.random.split(key)
    batch_idx = jax.random.permutation(subkey, n)[:batch_size]
    return batch_idx, key
