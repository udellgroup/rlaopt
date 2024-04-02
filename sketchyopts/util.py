import jax
import jax.numpy as jnp
from optax._src.base import (
    GradientTransformation,
    GradientTransformationExtraArgs,
    with_extra_args_support,
    ScalarOrSchedule,
)
from optax._src.transform import scale, scale_by_schedule, ScaleState

import abc
from typing import Callable, NamedTuple, Union, Optional
from jax.typing import ArrayLike
from jax import Array


class LinearOperator:
    r"""Base interface for abstract linear operators."""

    def __init__(self, shape: tuple, ndim: int):
        r"""Initialize the linear operator.

        Args:
          shape: Shape of the linear operator.
          ndim: Dimension of the linear operator.

        """
        self.shape = shape
        self.ndim = ndim

    @abc.abstractmethod
    def matmul(self, other: ArrayLike) -> Array:
        r"""Compute a matrix-vector or matrix-matrix product between this operator and a JAX array.

        Args:
          other: JAX array with matching dimension.

        Returns:
          A JAX array representing the resulting vector or matrix.

        """

    def __matmul__(self, other: ArrayLike) -> Array:
        r"""An alias for function :func:`sketchyopts.util.LinearOperator.matmul`.

        This overwrites the ``@`` operator.

        """
        return self.matmul(other)


class HessianLinearOperator(LinearOperator):
    def __init__(self, f, params, **f_extra_args):
        params_dim = jnp.ndim(params)
        params_shape = jnp.shape(params)

        f_partial = lambda x: f(x, **f_extra_args)
        self.jvp_func = lambda v: jax.jvp(jax.grad(f_partial), (params,), (v,))[1]

        super().__init__(shape=params_shape * 2, ndim=params_dim * 2)

    def matmul(self, other):
        if jnp.ndim(other) == 1:
            return self.jvp_func(other)
        elif jnp.ndim(other) == 2:
            return jax.vmap(self.jvp_func, 1, 1)(other)
        else:
            raise ValueError(
                "matmul input operand 1 must have ndim 1 or 2, but it has ndim {}".format(
                    jnp.ndim(other)
                )
            )


def shareble_state_named_chain(
    *transforms: tuple[str, GradientTransformationExtraArgs]
) -> GradientTransformationExtraArgs:
    """Chains optax gradient transformations with a sharing state.

    This extends the Optax `named_chain() <https://optax.readthedocs.io/en/latest/api/combining_optimizers.html>`_ by making state of a transformations available to all the succeeding transformations in the chain.

    The visibility of the chain state is realized by providing the chain state as the third argument to the update function of each transformation. This allows transformation to access the updated state of proceeding transformations as the chain is putting together the complete state by sequentially calling transformation update functions. This also means the transformation must all be `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ objects themselves.

    Adapted original Optax ``named_chain()`` docstring below.

    The `transforms` are ``(name, transformation)`` pairs, constituted of a string
    ``name`` and an associated gradient transformation ``transformation``. The
    gradient transformation must be an instance of ``GradientTransformationExtraArgs``.

    Each ``name`` is used as key for the state of the corresponding transformation
    within the ``named_chain`` state. Thus the state of the gradient transformation
    with a given ``name`` can be retrieved as ``opt_state[name]``.

    Example:

      .. code-block:: python

        # tx1 is a GradientTransformationExtraArgs with no extra_args.
        # tx2 is a GradientTransformationExtraArgs that requires `loss`.
        # tx3 is a GradientTransformationExtraArgs that requires `temperature`.

        tx = named_chain(('one', tx1), ('two', tx2), ('three', tx3))
        extra_args={'loss': 0.3, 'temperature': 0.01}
        tx.init(params)
        tx.update(grads, state, params, **extra_args)

    Args:
      transforms: An arbitrary number of ``(name, tx)`` pairs, constituted of a string ``name`` and an associated gradient transformation ``tx``. The latter is a `GradientTransformationExtraArgs`.

    Returns:
      A `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ object.
    """

    names = [name for name, _ in transforms]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Named transformations must have unique names, but got {names}"
        )

    transforms = [(name, with_extra_args_support(t)) for name, t in transforms]  # type: ignore

    def init_fn(params):
        states = {}
        for name, tx in transforms:
            states[name] = tx.init(params)
        return states

    def update_fn(updates, state, params=None, **extra_args):
        new_state = {}
        for name, tx in transforms:
            updates, new_state[name] = tx.update(
                updates, state[name], params, chain_state=new_state, **extra_args
            )
        return updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


# TODO: make this compatible with schedule
def scale_by_ref_learning_rate(
    learning_rate: Optional[float] = None,
    ref_state: Optional[str] = None,
    *,
    flip_sign: bool = True,
) -> GradientTransformationExtraArgs:
    r"""Scale by the (referenced) learning rate (either as scalar or as schedule).

    This function extends the Optax `scale_by_learning_rate() <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_learning_rate>`_ by adding an additional ``ref_state`` argument to allow learning rate to be defined in the referenced state of a named chain.

    Args:
      learning_rate: A scalar corresponding to a scaling factor for updates.
      ref_state: The name of the state in a named chain that provides the learning rate.
      flip_sign: Whether or not to flip the sign of the update (default ``True``). When set to ``True``, the function scales the update by the negative learning rate.

    Returns:
      A `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ object.
    """

    def init_fn(params):
        del params
        return ScaleState()

    def update_fn(updates, state, params=None, chain_state=None):
        del params
        # determine sign
        m = -1 if flip_sign else 1
        # use specified learning_rate if exists otherwise use the learning_rate from the referenced chain state
        lr = (
            learning_rate
            if learning_rate is not None
            else chain_state[ref_state].learning_rate  # type: ignore
        )
        # make update
        updates = jax.tree_util.tree_map(lambda g: m * lr * g, updates)
        return updates, state

    return GradientTransformationExtraArgs(init_fn, update_fn)
