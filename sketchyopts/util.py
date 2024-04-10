import jax
import jax.numpy as jnp
from optax._src.base import (
    GradientTransformation,
    GradientTransformationExtraArgs,
    ScalarOrSchedule,
    OptState,
    Params,
    Updates,
)
from optax._src.transform import scale, scale_by_learning_rate, ScaleByScheduleState
from optax._src import numerics

import abc
from typing import (
    Callable,
    NamedTuple,
    Union,
    Optional,
    Protocol,
    Any,
    Dict,
    Iterable,
    Mapping,
)
import inspect
from jax.typing import ArrayLike
from jax import Array
from chex import ArrayTree


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
        r"""Compute a matrix-vector or matrix-matrix product between this operator and
        a JAX array.

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


class TransformUpdateExtraArgsRefStateFn(Protocol):
    r"""An update function accepting chain state and additional keyword arguments."""

    @abc.abstractmethod
    def __call__(
        self,
        updates: Updates,
        state: OptState,
        params: Optional[Params] = None,
        chain_state: Optional[Dict[str, OptState]] = None,
        **extra_args: Any,
    ) -> tuple[Updates, OptState]:
        r"""Update function with chain state and optional extra arguments.

        For example, an update function that requires an additional loss parameter and
        relies on the state of preceding transformations in a named chain could be
        expressed as follows:

        >>> def update(updates, state, params=None, chain_state=None, *, loss, **extra_args):
        ...   del extra_args
        ...   # use loss value and chain_state
        ...   # access state of preceding transformation via chain_state[tx_name]

        Note that the loss value is keyword only, (it follows a ``*`` in the signature
        of the function). This implies users will get explicit errors if they try to use
        this gradient transformation without providing the required argument.

        Args:
          updates: The gradient updates passed to this transformation.
          state: The state associated with this transformation.
          params: Optional params.
          chain_state: The dictionary containing state of chained transformations.
            Also optional.
          **extra_args: Additional keyword arguments passed to this transform. All.
            implementors of this interface should accept arbitrary keyword arguments,
            ignoring those that are not needed for the current transformation.
            Transformations that require specific extra args should specify these
            using keyword-only arguments.

        Returns:
          Transformed updates, and an updated value for the state.
        """


class GradientTransformationExtraArgsRefState(GradientTransformationExtraArgs):
    """A specialization of GradientTransformationExtraArgs that supports chain state.

    Extends the existing GradientTransformationExtraArgs interface by adding
    support for passing chain state (``chain_state``) to the update function.

    By accepting but not using (ignoring) the chain state in the update function,
    any gradient transformation of type
    `GradientTransformation <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformation>`_
    and
    `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_
    can be converted to the one that supports chain state. Such conversion is
    realized by :func:`sketchyopts.util.with_ref_state_support`.

    Attributes:
      update: Overrides the type signature of the update in the base type to
        accept chain state.
    """

    update: TransformUpdateExtraArgsRefStateFn  # type: ignore


def with_ref_state_support(
    tx: GradientTransformation,
) -> GradientTransformationExtraArgsRefState:
    r"""Wraps a gradient transformation so that the update function takes chain state.

    This is analogous to
    `with_extra_args_support() <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.with_extra_args_support>`_.
    Main use of the function is to convert any gradient transformation to an instance of
    :class:`sketchyopts.util.GradientTransformationExtraArgsRefState`. This allows the
    :func:`sketchyopts.util.shareble_state_named_chain` to work with any transformation
    types.

    The wrapped transformation has its update function compatible with chain state
    (``chain_state``) and variable length argument. Any extra argument it previously
    takes is also preserved.

    Args:
      tx: A gradient transformation.

    Returns:
      A :class:`sketchyopts.util.GradientTransformationExtraArgsRefState` object.
    """

    # determine if update function accepts variadic arguments
    sig = inspect.signature(tx.update)
    params = list(sig.parameters.values())
    has_extra_args = params[-1].kind == inspect.Parameter.VAR_KEYWORD
    # obtain argument names and default values
    arg_names = [p.name for i, p in enumerate(params)]
    arg_vals = [p.default for i, p in enumerate(params)]

    # when the transformation is a GradientTransformationExtraArgsRefState object
    if isinstance(tx, GradientTransformationExtraArgsRefState):
        # no need to change the update function
        if has_extra_args:
            return tx
        # append variadic arguments to the existing update function signature
        else:
            arg_names = arg_names[4:]
            arg_vals = arg_vals[4:]

            def update(updates, state, params=None, chain_state=None, **extra_args):
                args = {
                    n: extra_args.get(n, v)
                    for n, v in zip(arg_names, arg_vals)
                    if (n in extra_args.keys()) or (v != inspect._empty)
                }
                return tx.update(updates, state, params, chain_state, **args)

            return GradientTransformationExtraArgsRefState(tx.init, update)

    # when the transformation is a GradientTransformationExtraArgs object
    if isinstance(tx, GradientTransformationExtraArgs):
        # add chain state argument to update function
        if has_extra_args:

            def update(updates, state, params=None, chain_state=None, **extra_args):
                del chain_state
                return tx.update(updates, state, params, **extra_args)

            return GradientTransformationExtraArgsRefState(tx.init, update)
        # append variadic arguments to the existing update function signature
        else:
            arg_names = arg_names[3:]
            arg_vals = arg_vals[3:]

            def update(updates, state, params=None, chain_state=None, **extra_args):
                del chain_state
                args = {
                    n: extra_args.get(n, v)
                    for n, v in zip(arg_names, arg_vals)
                    if (n in extra_args.keys()) or (v != inspect._empty)
                }
                return tx.update(updates, state, params, **args)

            return GradientTransformationExtraArgsRefState(tx.init, update)

    # when the transformation is a GradientTransformation object
    def update(updates, state, params=None, chain_state=None, **extra_args):
        del chain_state
        del extra_args
        return tx.update(updates, state, params)

    return GradientTransformationExtraArgsRefState(tx.init, update)


def shareble_state_named_chain(
    *transforms: tuple[str, GradientTransformation]
) -> GradientTransformationExtraArgs:
    r"""Chains optax gradient transformations with a sharing state.

    This extends the Optax
    `named_chain() <https://optax.readthedocs.io/en/latest/api/combining_optimizers.html>`_
    by making state of a transformations available to all the succeeding transformations
    in the chain.

    The visibility of the chain state is realized by providing the chain state
    (``chain_state``) as the third argument to the update function of each
    transformation. This allowstransformation to access the updated state of proceeding
    transformations as the chain is putting together the complete state by sequentially
    calling transformation update functions. This also means each transformation must
    be an object of one of the following:

    - `GradientTransformation <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformation>`_,
    - `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_,
    - or :class:`sketchyopts.util.GradientTransformationExtraArgsRefState`

    ----

    Adapted original Optax ``named_chain()`` docstring below.

        The `transforms` are ``(name, transformation)`` pairs, constituted of a string
        ``name`` and an associated gradient transformation ``transformation``. The
        gradient transformation must be an instance of ``GradientTransformation``,
        ``GradientTransformationExtraArgs``, or
        ``GradientTransformationExtraArgsRefState``.

        Each ``name`` is used as key for the state of the corresponding transformation
        within the ``named_chain`` state. Thus the state of the gradient
        transformation with a given ``name`` can be retrieved as ``opt_state[name]``.

    Example:

      .. code-block:: python

        # tx1 is a GradientTransformation with no extra_args.
        # tx2 is a GradientTransformationExtraArgs that requires `loss`.
        # tx3 is a GradientTransformationExtraArgsRefState that requires `temperature`
        #          and uses information from the state of tx2.
        #          This can be done by passing the name of tx2 as a string to the
        #          transformation at its instantiation (depending on implementation).
        #          e.g. tx3 = transform_fn_name(some_args, ref_state='two')

        tx = shareble_state_named_chain(('one', tx1), ('two', tx2), ('three', tx3))
        extra_args={'loss': 0.3, 'temperature': 0.01}
        tx.init(params)
        tx.update(grads, state, params, **extra_args)

    Args:
      transforms: An arbitrary number of ``(name, tx)`` pairs, constituted of a string
        ``name`` and an associated gradient transformation ``tx``. The latter is one of
        `GradientTransformation`, `GradientTransformationExtraArgs`, or
        `GradientTransformationExtraArgsRefState`.

    Returns:
      A `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ object.
    """

    names = [name for name, _ in transforms]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Named transformations must have unique names, but got {names}"
        )

    transforms = [(name, with_ref_state_support(t)) for name, t in transforms]  # type: ignore

    def init_fn(params):
        states = {}
        for name, tx in transforms:
            states[name] = tx.init(params)
        return states

    def update_fn(updates, state, params=None, **extra_args):
        new_state = {}
        for name, tx in transforms:
            updates, new_state[name] = tx.update(
                updates, state[name], params, chain_state=new_state, **extra_args  # type: ignore
            )
        return updates, new_state

    return GradientTransformationExtraArgs(init_fn, update_fn)


def scale_by_ref_learning_rate(
    learning_rate: Optional[ScalarOrSchedule] = None,
    ref_state: Optional[str] = None,
    *,
    flip_sign: bool = True,
) -> GradientTransformation:
    r"""Scales by the (referenced) learning rate (either as scalar or as schedule).

    This function extends the Optax
    `scale_by_learning_rate() <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.scale_by_learning_rate>`_
    by adding an additional ``ref_state`` argument to allow learning rate to be defined
    in the referenced state of a named chain.

    Args:
      learning_rate: Can either be a scalar or a schedule (*i.e.* a callable that maps
        an (int) step to a float).
      ref_state: The name of the state in a named chain that provides the learning rate.
      flip_sign: Whether or not to flip the sign of the update (default ``True``). When
        set to ``True``, the function scales the update by the negative learning rate.

    Returns:
      A `GradientTransformationExtraArgs <https://optax.readthedocs.io/en/latest/api/transformations.html#optax.GradientTransformationExtraArgs>`_ object.
    """

    if (learning_rate is None) and (ref_state is None):
        raise ValueError("learning_rate and ref_state cannot both be None")

    if learning_rate is not None:
        return scale_by_learning_rate(
            learning_rate=learning_rate,
            flip_sign=flip_sign,
        )

    def init_fn(params):
        del params
        return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None, chain_state=None):
        del params
        # validate referenced learning rate
        if (chain_state is None) or (ref_state not in chain_state.keys()):
            raise ValueError("ref_state does not exist")
        if "learning_rate" not in dir(chain_state[ref_state]):
            raise ValueError(
                "learning_rate is not an attribute of the referenced state"
            )
        if (not callable(chain_state[ref_state].learning_rate)) and (not isinstance(chain_state[ref_state].learning_rate, ScalarOrSchedule)):  # type: ignore
            raise ValueError("learning rate has incompatible type")
        # determine sign
        m = -1 if flip_sign else 1
        # use the learning_rate from the referenced chain state
        if callable(chain_state[ref_state].learning_rate):
            step_size = chain_state[ref_state].learning_rate(state.count)
        else:
            step_size = chain_state[ref_state].learning_rate
        # make update
        updates = jax.tree_util.tree_map(lambda g: m * step_size * g, updates)

        return updates, ScaleByScheduleState(count=numerics.safe_int32_increment(state.count))  # type: ignore

    return GradientTransformationExtraArgsRefState(init_fn, update_fn)
