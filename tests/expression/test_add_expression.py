"""Tests for AddExpression class."""

import pytest
import torch

from rlaopt.expression import Expression, Variable
from rlaopt.expression.op_expressions import AddExpression
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def smooth_expr():
    """Create a smooth expression (Variable)."""
    x = Variable((5,), name="x")
    x.value.data = torch.ones(5)
    return x


@pytest.fixture
def another_smooth_expr():
    """Create another smooth expression."""
    y = Variable((5,), name="y")
    y.value.data = torch.ones(5) * 2
    return y


@pytest.fixture
def non_smooth_expr():
    """Create a non-smooth proxable expression mock."""

    class MockNonSmooth(Expression):
        def __init__(self):
            super().__init__()

        def is_smooth(self):
            return False

        def is_proxable(self):
            return True

        def prox(self, location, prox_scaling):
            return location * 0.5

        def forward(self):
            return torch.ones(5) * 3

    return MockNonSmooth()


@pytest.fixture
def non_proxable_expr():
    """Create a non-smooth non-proxable expression mock."""

    class MockNonProxable(Expression):
        def __init__(self):
            super().__init__()

        def is_smooth(self):
            return False

        def is_proxable(self):
            return False

        def forward(self):
            return torch.ones(5) * 4

    return MockNonProxable()


class TestAddExpression:
    """Test AddExpression concrete implementation."""

    # ----------------------
    # Initialization and basic operations
    # ----------------------

    def test_init_and_basic_properties(self, smooth_expr, another_smooth_expr):
        """Test initialization and basic expression properties."""
        add_expr = AddExpression(smooth_expr, another_smooth_expr)
        assert add_expr.n_exprs == 2
        assert len(list(add_expr.exprs)) == 2

    def test_op_method_with_various_inputs(self, smooth_expr):
        """Test op() method with single, two, and multiple expressions."""
        # Single expression
        single_expr = AddExpression(smooth_expr)
        assert torch.equal(single_expr.op([torch.ones(5) * 5]), torch.ones(5) * 5)

        # Two expressions
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        double_expr = AddExpression(x, y)
        assert torch.equal(
            double_expr.op([torch.ones(5), torch.ones(5) * 2]), torch.ones(5) * 3
        )

        # Multiple expressions (nested)
        z = Variable((3,), name="z")
        multi_expr = AddExpression(AddExpression(x, y), z)
        values = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor([7.0, 8.0, 9.0]),
        ]
        assert torch.equal(multi_expr.op(values), torch.tensor([12.0, 15.0, 18.0]))

    # ----------------------
    # Smoothness and proxability tests
    # ----------------------

    def test_proxability_with_different_term_combinations(
        self, smooth_expr, another_smooth_expr, non_smooth_expr, non_proxable_expr
    ):
        """Test proxability rules for different expression combinations."""
        # All smooth expressions are proxable
        all_smooth = AddExpression(smooth_expr, another_smooth_expr)
        assert all_smooth.is_proxable() is True
        assert all_smooth.num_non_smooth_exprs == 0

        # One non-smooth proxable term is proxable
        one_non_smooth = AddExpression(smooth_expr, non_smooth_expr)
        assert one_non_smooth.is_proxable() is True
        assert one_non_smooth.num_non_smooth_exprs == 1

        # Non-proxable term makes sum non-proxable
        with_non_proxable = AddExpression(smooth_expr, non_proxable_expr)
        assert with_non_proxable.is_proxable() is False

    def test_overlapping_parameters_prevents_proxability(self):
        """Test non-smooth terms with overlapping parameters are not proxable."""
        x = Variable((5,), name="x")

        class MockNonSmoothWithParam(Expression):
            def __init__(self, var):
                super().__init__()
                self.var = var

            def is_smooth(self):
                return False

            def is_proxable(self):
                return True

            def forward(self):
                return self.var.forward()

        expr1 = MockNonSmoothWithParam(x)
        expr2 = MockNonSmoothWithParam(x)
        add_expr = AddExpression(expr1, expr2)

        assert add_expr.is_proxable() is False

    # ----------------------
    # Operator splitting tests
    # ----------------------

    def test_operator_split_scenarios(self):
        """Test operator_split with all smooth, all non-smooth, and mixed."""
        # All smooth
        x1 = Variable((5,), name="x1")
        y1 = Variable((5,), name="y1")
        all_smooth = AddExpression(x1, y1)
        smooth, non_smooth = all_smooth.operator_split()
        assert smooth is not None and smooth.n_exprs == 2
        assert non_smooth is None

        # All non-smooth
        class MockNonSmooth(Expression):
            def __init__(self):
                super().__init__()

            def is_smooth(self):
                return False

            def is_proxable(self):
                return True

            def prox(self, location, prox_scaling):  # <-- ADD THIS
                return location * 0.5

            def forward(self):
                return torch.ones(5)

        class MockNonProxable(Expression):
            def __init__(self):
                super().__init__()

            def is_smooth(self):
                return False

            def is_proxable(self):
                return False

            def forward(self):
                return torch.ones(5)

        ns1 = MockNonSmooth()
        ns2 = MockNonProxable()
        all_non_smooth = AddExpression(ns1, ns2)
        smooth, non_smooth = all_non_smooth.operator_split()
        assert smooth is None
        assert non_smooth is not None and non_smooth.n_exprs == 2

        # Mixed
        x2 = Variable((5,), name="x2")
        ns3 = MockNonSmooth()
        mixed = AddExpression(x2, ns3)
        smooth, non_smooth = mixed.operator_split()
        assert smooth is not None and smooth.n_exprs == 1
        assert non_smooth is not None and non_smooth.n_exprs == 1
        assert smooth.is_smooth() is True
        assert non_smooth.is_smooth() is False

    # ----------------------
    # Proximal operator tests
    # ----------------------

    def test_prox_behavior_based_on_proxability(
        self, smooth_expr, non_smooth_expr, non_proxable_expr
    ):
        """Test prox() behavior for different proxability scenarios."""
        # Raises error when not proxable
        non_prox_sum = AddExpression(smooth_expr, non_proxable_expr)
        with pytest.raises(NotImplementedError, match="not proxable"):
            non_prox_sum.prox(torch.ones(5), 1.0)

        # Single non-smooth term delegates correctly
        single_non_smooth = AddExpression(smooth_expr, non_smooth_expr)
        location = torch.ones(5) * 10
        result = single_non_smooth.prox(location, 1.0)
        assert torch.equal(
            result, torch.ones(5) * 5
        )  # MockNonSmooth.prox returns location * 0.5

        # Multiple non-smooth terms use dict interface
        class MockNonSmoothSeparate(Expression):
            def __init__(self, name):
                super().__init__()
                self.param_name = name

            def is_smooth(self):
                return False

            def is_proxable(self):
                return True

            def prox(self, location, prox_scaling):
                return location * 0.5

            def forward(self):
                return torch.ones(3)

        expr1 = MockNonSmoothSeparate("expr1")
        expr2 = MockNonSmoothSeparate("expr2")
        multi_non_smooth = AddExpression(expr1, expr2)

        location_dict = TensorDict(
            {"param1": torch.ones(3) * 10, "param2": torch.ones(3) * 20}
        )
        result = multi_non_smooth.prox(location_dict, 1.0)

        assert isinstance(result, TensorDict)
        assert torch.equal(result["param1"], torch.ones(3) * 5)
        assert torch.equal(result["param2"], torch.ones(3) * 10)
