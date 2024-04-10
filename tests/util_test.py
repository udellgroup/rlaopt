import jax.random
import jax.numpy as jnp
import optax
import pytest
import chex

from sketchyopts.util import (
    GradientTransformationExtraArgsRefState,
    shareble_state_named_chain,
    with_ref_state_support,
    scale_by_ref_learning_rate,
)
from optax._src.transform import scale_by_adam, trace, scale_by_learning_rate
from optax._src.base import GradientTransformation, GradientTransformationExtraArgs
from optax._src.combine import chain
from collections import namedtuple


class TestRefState:

    def test_ref_state_arg_chaining(self):
        """
        Test chaining function (i.e. shareble_state_named_chain).
        We expect the chain state of the current iteration to be visible to all transformations.
        We also verify extra arguments get passed through transformations correctly.
        """
        b1 = 0.9
        b2 = 0.999
        lr = 0.0
        params = {"a": 1, "b": 2}

        def init_fn(params):
            del params
            return tuple()

        def update_fn_1(updates, state, params=None, chain_state=None, **extra_args):
            del params, extra_args
            assert chain_state == {
                "adam": namedtuple("ScaleByAdamState", ["count", "mu", "nu"])(
                    jnp.ones([], jnp.int32),
                    {"a": jnp.array((1 - b1) * 1), "b": jnp.array((1 - b1) * 2)},
                    {"a": jnp.array((1 - b2) * 1), "b": jnp.array((1 - b2) * 4)},
                )
            }
            return updates, state

        def update_fn_2(
            updates, state, params=None, chain_state=None, *, tx2_arg, **extra_args
        ):
            del params
            assert tx2_arg == "some string"
            assert chain_state == {
                "adam": namedtuple("ScaleByAdamState", ["count", "mu", "nu"])(
                    jnp.ones([], jnp.int32),
                    {"a": jnp.array((1 - b1) * 1), "b": jnp.array((1 - b1) * 2)},
                    {"a": jnp.array((1 - b2) * 1), "b": jnp.array((1 - b2) * 4)},
                ),
                "tx1": tuple(),
                "schedule": namedtuple("ScaleByScheduleState", ["count"])(
                    jnp.ones([], jnp.int32)
                ),
            }
            return updates, state

        tx1 = GradientTransformationExtraArgsRefState(init_fn, update_fn_1)
        tx2 = GradientTransformationExtraArgsRefState(init_fn, update_fn_2)
        chain = shareble_state_named_chain(
            ("adam", scale_by_adam(b1, b2)),
            ("tx1", tx1),
            ("schedule", scale_by_learning_rate(lambda count: lr)),
            ("tx2", tx2),
        )

        state = chain.init(params)
        chain.update(
            params, state, tx2_arg="some string", ignore_arg="expect to have no effect"
        )

    def test_with_ref_state_support(self):
        """
        Test the function that augments any transformation to GradientTransformationExtraArgsRefState object.
        Function with_ref_state_support should be able to handle any update function signature.
        """

        def init_fn(params):
            del params
            return tuple()

        def update_fn(updates, state, params=None):
            assert params is not None
            return updates, state

        def update_fn_with_extra_args_1(
            updates, state, params=None, *, required_kwarg_1
        ):
            assert required_kwarg_1 == "hello"
            assert params is not None
            return updates, state

        def update_fn_with_extra_args_2(
            updates, state, params=None, required_kwarg_1="hey", **extra_args
        ):
            del extra_args
            assert required_kwarg_1 == "hello"
            assert params is not None
            return updates, state

        def update_fn_with_extra_args_ref_state_1(
            updates,
            state,
            params=None,
            chain_state=None,
            *,
            required_kwarg_1,
            required_kwarg_2,
        ):
            assert required_kwarg_1 == "hello"
            assert required_kwarg_2 == "hi"
            assert params is not None
            assert chain_state is not None
            return updates, state

        def update_fn_with_extra_args_ref_state_2(
            updates,
            state,
            params=None,
            chain_state=None,
            required_kwarg_1="hey",
            required_kwarg_2="hey",
            **extra_args,
        ):
            del extra_args
            assert required_kwarg_1 == "hello"
            assert required_kwarg_2 == "hi"
            assert params is not None
            assert chain_state is not None
            return updates, state

        def update_fn_with_extra_args_ref_state_3(
            updates, state, params=None, chain_state=None, **extra_args
        ):
            assert extra_args is not None
            assert params is not None
            assert chain_state is not None
            return updates, state

        tx1 = GradientTransformation(init_fn, update_fn)
        tx1_added_support = with_ref_state_support(tx1)
        tx2 = GradientTransformationExtraArgs(init_fn, update_fn_with_extra_args_1)
        tx2_added_support = with_ref_state_support(tx2)
        tx3 = GradientTransformationExtraArgs(init_fn, update_fn_with_extra_args_2)
        tx3_added_support = with_ref_state_support(tx3)
        tx4 = GradientTransformationExtraArgsRefState(
            init_fn, update_fn_with_extra_args_ref_state_1
        )
        tx5 = GradientTransformationExtraArgsRefState(
            init_fn, update_fn_with_extra_args_ref_state_2
        )
        tx6 = GradientTransformationExtraArgsRefState(
            init_fn, update_fn_with_extra_args_ref_state_3
        )

        chain = shareble_state_named_chain(
            ("tx1", tx1),
            ("tx1_added_support", tx1_added_support),
            ("tx2", tx2),
            ("tx2_added_support", tx2_added_support),
            ("tx3", tx3),
            ("tx3_added_support", tx3_added_support),
            ("tx4", tx4),
            ("tx5", tx5),
            ("tx6", tx6),
        )

        # check object class
        assert not isinstance(tx1, GradientTransformationExtraArgsRefState)
        assert not isinstance(tx2, GradientTransformationExtraArgsRefState)
        assert not isinstance(tx3, GradientTransformationExtraArgsRefState)

        assert isinstance(tx1_added_support, GradientTransformation)
        assert isinstance(tx1_added_support, GradientTransformationExtraArgs)
        assert isinstance(tx1_added_support, GradientTransformationExtraArgsRefState)

        assert isinstance(tx2_added_support, GradientTransformation)
        assert isinstance(tx2_added_support, GradientTransformationExtraArgs)
        assert isinstance(tx2_added_support, GradientTransformationExtraArgsRefState)

        assert isinstance(tx3_added_support, GradientTransformation)
        assert isinstance(tx3_added_support, GradientTransformationExtraArgs)
        assert isinstance(tx3_added_support, GradientTransformationExtraArgsRefState)

        assert isinstance(tx4, GradientTransformation)
        assert isinstance(tx4, GradientTransformationExtraArgs)

        assert isinstance(tx5, GradientTransformation)
        assert isinstance(tx5, GradientTransformationExtraArgs)

        assert isinstance(tx6, GradientTransformation)
        assert isinstance(tx6, GradientTransformationExtraArgs)

        assert isinstance(chain, GradientTransformation)
        assert isinstance(chain, GradientTransformationExtraArgs)
        assert not isinstance(chain, GradientTransformationExtraArgsRefState)

        # check augmented transformations in the chain
        params = {"a": 1, "b": 2}
        state = chain.init(params)
        chain.update(
            params,
            state,
            params,
            required_kwarg_1="hello",
            required_kwarg_2="hi",
            ignored_kwarg="howdy",
        )


class TestScaleByRefLearningRate:

    @pytest.fixture(scope="class")
    def test_parameters(self):
        """
        Define test parameters.
        """
        num_steps = 50
        lr = 1e-2
        schedule = lambda k: 1 / (k + 1)
        params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

        return namedtuple(
            "TestParameters",
            ["num_steps", "lr", "schedule", "params", "per_step_updates"],
        )(num_steps, lr, schedule, params, per_step_updates)

    @pytest.fixture(scope="class")
    def pass_lr_transformation(self):
        """
        Define a transformation that stores scalar learning rate or schedule (that can be referenced to).
        """

        def pass_learning_rate(learning_rate):
            def init_fn(params):
                del params
                return namedtuple("LRState", ["learning_rate"])(learning_rate)

            def update_fn(updates, state, params=None):
                del params
                return updates, state

            return GradientTransformation(init_fn, update_fn)

        return pass_learning_rate

    @pytest.fixture(scope="class")
    def test_benchmarks(self, test_parameters):
        """
        Compute the expected results using only built-in transformations of Optax.
        """
        transformations_lr = [
            scale_by_adam(),
            trace(decay=0, nesterov=False),
            scale_by_learning_rate(test_parameters.lr),
        ]

        transformations_schedule = [
            scale_by_adam(),
            trace(decay=0, nesterov=False),
            scale_by_learning_rate(test_parameters.schedule),
        ]

        chain_lr = chain(*transformations_lr)
        chain_schedule = chain(*transformations_schedule)

        chain_params_lr = test_parameters.params
        chain_params_schedule = test_parameters.params

        state_lr = chain_lr.init(chain_params_lr)
        state_schedule = chain_schedule.init(chain_params_schedule)

        for _ in range(test_parameters.num_steps):
            updates_lr, state_lr = chain_lr.update(
                test_parameters.per_step_updates, state_lr
            )
            updates_schedule, state_schedule = chain_schedule.update(
                test_parameters.per_step_updates, state_schedule
            )

            chain_params_lr = optax.apply_updates(chain_params_lr, updates_lr)
            chain_params_schedule = optax.apply_updates(
                chain_params_schedule, updates_schedule
            )

        return namedtuple("TestBenchmarks", ["lr", "schedule"])(
            chain_params_lr, chain_params_schedule
        )

    def test_specified_learning_rate(self, test_parameters, test_benchmarks):
        """
        Verify scale_by_ref_learning_rate works correctly with manually specified scalar learning rate.
        """
        transformations = [
            ("adam", scale_by_adam()),
            ("momentum", trace(decay=0, nesterov=False)),
            ("scaling", scale_by_ref_learning_rate(learning_rate=test_parameters.lr)),
        ]
        named_chain = shareble_state_named_chain(*transformations)

        named_chain_params = test_parameters.params
        state = named_chain.init(named_chain_params)

        for _ in range(test_parameters.num_steps):
            updates, state = named_chain.update(test_parameters.per_step_updates, state)
            named_chain_params = optax.apply_updates(named_chain_params, updates)

        chex.assert_trees_all_close(named_chain_params, test_benchmarks.lr)

    def test_specified_schedule(self, test_parameters, test_benchmarks):
        """
        Verify scale_by_ref_learning_rate works correctly with manually specified schedule.
        """
        transformations = [
            ("adam", scale_by_adam()),
            ("momentum", trace(decay=0, nesterov=False)),
            (
                "scaling",
                scale_by_ref_learning_rate(learning_rate=test_parameters.schedule),
            ),
        ]
        named_chain = shareble_state_named_chain(*transformations)

        named_chain_params = test_parameters.params
        state = named_chain.init(named_chain_params)

        for _ in range(test_parameters.num_steps):
            updates, state = named_chain.update(test_parameters.per_step_updates, state)
            named_chain_params = optax.apply_updates(named_chain_params, updates)

        chex.assert_trees_all_close(named_chain_params, test_benchmarks.schedule)

    def test_ref_learning_rate(
        self, test_parameters, test_benchmarks, pass_lr_transformation
    ):
        """
        Verify scale_by_ref_learning_rate works correctly with referenced scalar learning rate.
        """
        transformations = [
            ("adam", scale_by_adam()),
            ("momentum", trace(decay=0, nesterov=False)),
            ("lr", pass_lr_transformation(test_parameters.lr)),
            ("scaling", scale_by_ref_learning_rate(ref_state="lr")),
        ]
        named_chain = shareble_state_named_chain(*transformations)

        named_chain_params = test_parameters.params
        state = named_chain.init(named_chain_params)

        for _ in range(test_parameters.num_steps):
            updates, state = named_chain.update(test_parameters.per_step_updates, state)
            named_chain_params = optax.apply_updates(named_chain_params, updates)

        chex.assert_trees_all_close(named_chain_params, test_benchmarks.lr)

    def test_ref_schedule(
        self, test_parameters, test_benchmarks, pass_lr_transformation
    ):
        """
        Verify scale_by_ref_learning_rate works correctly with referenced schedule.
        """
        transformations = [
            ("adam", scale_by_adam()),
            ("momentum", trace(decay=0, nesterov=False)),
            ("schedule", pass_lr_transformation(test_parameters.schedule)),
            ("scaling", scale_by_ref_learning_rate(ref_state="schedule")),
        ]
        named_chain = shareble_state_named_chain(*transformations)

        named_chain_params = test_parameters.params
        state = named_chain.init(named_chain_params)

        for _ in range(test_parameters.num_steps):
            updates, state = named_chain.update(test_parameters.per_step_updates, state)
            named_chain_params = optax.apply_updates(named_chain_params, updates)

        chex.assert_trees_all_close(named_chain_params, test_benchmarks.schedule)
