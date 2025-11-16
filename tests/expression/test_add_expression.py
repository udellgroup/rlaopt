"""Tests for AddExpression class."""

import pytest
import torch

from rlaopt.expression import Expression, Variable
from rlaopt.expression.op_expressions import AddExpression


@pytest.fixture
def smooth_expr():
    """Create a smooth expression (Variable)."""
    return Variable(torch.ones(5), name="x")


@pytest.fixture
def another_smooth_expr():
    """Create another smooth expression."""
    return Variable(torch.ones(5) * 2, name="y")


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

        def tree(self):
            raise NotImplementedError()

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

        def tree(self):
            raise NotImplementedError()

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
    # Smooth/non-smooth partitioning tests
    # ----------------------

    def test_get_smooth_part_mixed_terms(self, smooth_expr, non_smooth_expr):
        """Test get_smooth_part returns only smooth terms."""
        mixed = AddExpression(smooth_expr, non_smooth_expr)

        smooth_part = mixed.get_smooth_part()

        assert smooth_part is not None
        assert smooth_part.is_smooth() is True
        assert smooth_part.n_exprs == 1
        # Verify it evaluates to the smooth term's value
        assert torch.equal(smooth_part.forward(), smooth_expr.forward())

    def test_get_non_smooth_exprs_mixed_terms(self, smooth_expr, non_smooth_expr):
        """Test get_non_smooth_exprs returns only non-smooth terms as a list."""
        mixed = AddExpression(smooth_expr, non_smooth_expr)

        non_smooth_exprs = mixed.get_non_smooth_exprs()

        assert isinstance(non_smooth_exprs, list)
        assert len(non_smooth_exprs) == 1
        assert non_smooth_exprs[0].is_smooth() is False
        # Verify it evaluates to the non-smooth term's value
        assert torch.equal(non_smooth_exprs[0].forward(), non_smooth_expr.forward())

    def test_get_smooth_part_all_smooth(self, smooth_expr, another_smooth_expr):
        """Test get_smooth_part with all smooth terms returns all terms."""
        all_smooth = AddExpression(smooth_expr, another_smooth_expr)

        smooth_part = all_smooth.get_smooth_part()

        assert smooth_part is not None
        assert smooth_part.n_exprs == 2
        assert smooth_part.is_smooth() is True
        # Should equal the original sum
        assert torch.equal(smooth_part.forward(), all_smooth.forward())

    def test_get_non_smooth_exprs_all_smooth(self, smooth_expr, another_smooth_expr):
        """Test get_non_smooth_exprs returns empty list when all terms are smooth."""
        all_smooth = AddExpression(smooth_expr, another_smooth_expr)

        non_smooth_exprs = all_smooth.get_non_smooth_exprs()

        assert isinstance(non_smooth_exprs, list)
        assert len(non_smooth_exprs) == 0

    def test_get_smooth_part_all_non_smooth(self, non_smooth_expr, non_proxable_expr):
        """Test get_smooth_part returns Constant(0.0) when all terms are non-smooth."""
        from rlaopt.expression.constant import Constant

        all_non_smooth = AddExpression(non_smooth_expr, non_proxable_expr)

        smooth_part = all_non_smooth.get_smooth_part()

        assert smooth_part is not None
        assert isinstance(smooth_part, Constant)
        assert smooth_part.is_smooth() is True
        assert torch.equal(smooth_part.forward(), torch.tensor(0.0))

    def test_get_non_smooth_exprs_all_non_smooth(
        self, non_smooth_expr, non_proxable_expr
    ):
        """Test get_non_smooth_exprs with all non-smooth returns all terms."""
        all_non_smooth = AddExpression(non_smooth_expr, non_proxable_expr)

        non_smooth_exprs = all_non_smooth.get_non_smooth_exprs()

        assert isinstance(non_smooth_exprs, list)
        assert len(non_smooth_exprs) == 2
        for expr in non_smooth_exprs:
            assert expr.is_smooth() is False

    def test_partition_completeness(self, smooth_expr, non_smooth_expr):
        """Test that smooth + non-smooth parts equal the original sum."""
        mixed = AddExpression(smooth_expr, non_smooth_expr)

        smooth_part = mixed.get_smooth_part()
        non_smooth_exprs = mixed.get_non_smooth_exprs()

        # Both parts should exist
        assert smooth_part is not None
        assert len(non_smooth_exprs) == 1

        # Sum of parts should equal original
        combined = smooth_part.forward() + non_smooth_exprs[0].forward()
        assert torch.equal(combined, mixed.forward())

    def test_multiple_smooth_and_non_smooth_terms(
        self, smooth_expr, another_smooth_expr, non_smooth_expr, non_proxable_expr
    ):
        """Test partitioning with multiple terms of each type."""
        mixed = AddExpression(
            smooth_expr, non_smooth_expr, another_smooth_expr, non_proxable_expr
        )

        smooth_part = mixed.get_smooth_part()
        non_smooth_exprs = mixed.get_non_smooth_exprs()

        # Should have 2 smooth terms
        assert smooth_part is not None
        assert smooth_part.n_exprs == 2
        assert smooth_part.is_smooth() is True

        # Should have 2 non-smooth terms
        assert len(non_smooth_exprs) == 2
        for expr in non_smooth_exprs:
            assert expr.is_smooth() is False

        # Verify sum
        non_smooth_sum = sum(expr.forward() for expr in non_smooth_exprs)
        combined = smooth_part.forward() + non_smooth_sum
        assert torch.equal(combined, mixed.forward())

    # ----------------------
    # Tree representation tests
    # ----------------------

    def test_tree_structure(self, smooth_expr, another_smooth_expr):
        """Test tree() returns correct structure."""
        from rlaopt.expression import ExprTree

        add_expr = AddExpression(smooth_expr, another_smooth_expr)

        expected = ExprTree(
            "AddExpression",
            smooth_expr.tree(),
            another_smooth_expr.tree(),
            is_commutative=True,
        )
        assert add_expr.tree() == expected
