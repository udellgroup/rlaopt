"""Tests for SumExpression class."""

import pytest
import torch

from rlaopt.expression import Expression, Variable
from rlaopt.expression.op_expressions import SumExpression


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
    """Create a non-smooth expression mock."""

    class MockNonSmooth(Expression):
        def __init__(self):
            super().__init__()

        def is_smooth(self):
            return False

        def forward(self):
            return torch.ones(5) * 3

        def tree(self):
            raise NotImplementedError()

    return MockNonSmooth()


@pytest.fixture
def another_non_smooth_expr():
    """Create another non-smooth expression mock."""

    class MockNonSmooth2(Expression):
        def __init__(self):
            super().__init__()

        def is_smooth(self):
            return False

        def forward(self):
            return torch.ones(5) * 4

        def tree(self):
            raise NotImplementedError()

    return MockNonSmooth2()


class TestSumExpression:
    """Test SumExpression concrete implementation."""

    # ----------------------
    # Initialization and basic operations
    # ----------------------

    def test_init_and_basic_properties(self, smooth_expr, another_smooth_expr):
        """Test initialization and basic expression properties."""
        add_expr = SumExpression(smooth_expr, another_smooth_expr)
        assert add_expr.n_exprs == 2
        assert len(list(add_expr.exprs)) == 2

    def test_op_method_with_various_inputs(self, smooth_expr):
        """Test op() method with single, two, and multiple expressions."""
        # Single expression
        single_expr = SumExpression(smooth_expr)
        assert torch.equal(single_expr.op([torch.ones(5) * 5]), torch.ones(5) * 5)

        # Two expressions
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        double_expr = SumExpression(x, y)
        assert torch.equal(
            double_expr.op([torch.ones(5), torch.ones(5) * 2]), torch.ones(5) * 3
        )

        # Multiple expressions (nested)
        z = Variable((3,), name="z")
        multi_expr = SumExpression(SumExpression(x, y), z)
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
        mixed = SumExpression(smooth_expr, non_smooth_expr)

        smooth_part = mixed.get_smooth_part()

        assert smooth_part is not None
        assert smooth_part.is_smooth() is True
        assert smooth_part.n_exprs == 1
        # Verify it evaluates to the smooth term's value
        assert torch.equal(smooth_part.forward(), smooth_expr.forward())

    def test_get_non_smooth_exprs_mixed_terms(self, smooth_expr, non_smooth_expr):
        """Test get_non_smooth_exprs returns only non-smooth terms as a list."""
        mixed = SumExpression(smooth_expr, non_smooth_expr)

        non_smooth_exprs = mixed.get_non_smooth_exprs()

        assert isinstance(non_smooth_exprs, list)
        assert len(non_smooth_exprs) == 1
        assert non_smooth_exprs[0].is_smooth() is False
        # Verify it evaluates to the non-smooth term's value
        assert torch.equal(non_smooth_exprs[0].forward(), non_smooth_expr.forward())

    def test_get_smooth_part_all_smooth(self, smooth_expr, another_smooth_expr):
        """Test get_smooth_part with all smooth terms returns all terms."""
        all_smooth = SumExpression(smooth_expr, another_smooth_expr)

        smooth_part = all_smooth.get_smooth_part()

        assert smooth_part is not None
        assert smooth_part.n_exprs == 2
        assert smooth_part.is_smooth() is True
        # Should equal the original sum
        assert torch.equal(smooth_part.forward(), all_smooth.forward())

    def test_get_non_smooth_exprs_all_smooth(self, smooth_expr, another_smooth_expr):
        """Test get_non_smooth_exprs returns empty list when all terms are smooth."""
        all_smooth = SumExpression(smooth_expr, another_smooth_expr)

        non_smooth_exprs = all_smooth.get_non_smooth_exprs()

        assert isinstance(non_smooth_exprs, list)
        assert len(non_smooth_exprs) == 0

    def test_get_smooth_part_all_non_smooth(
        self, non_smooth_expr, another_non_smooth_expr
    ):
        """Test get_smooth_part returns Constant(0.0) when all terms are non-smooth."""
        from rlaopt.expression.constant import Constant

        all_non_smooth = SumExpression(non_smooth_expr, another_non_smooth_expr)

        smooth_part = all_non_smooth.get_smooth_part()

        assert smooth_part is not None
        assert isinstance(smooth_part, Constant)
        assert smooth_part.is_smooth() is True
        assert torch.equal(smooth_part.forward(), torch.tensor(0.0))

    def test_get_non_smooth_exprs_all_non_smooth(
        self, non_smooth_expr, another_non_smooth_expr
    ):
        """Test get_non_smooth_exprs with all non-smooth returns all terms."""
        all_non_smooth = SumExpression(non_smooth_expr, another_non_smooth_expr)

        non_smooth_exprs = all_non_smooth.get_non_smooth_exprs()

        assert isinstance(non_smooth_exprs, list)
        assert len(non_smooth_exprs) == 2
        for expr in non_smooth_exprs:
            assert expr.is_smooth() is False

    def test_partition_completeness(self, smooth_expr, non_smooth_expr):
        """Test that smooth + non-smooth parts equal the original sum."""
        mixed = SumExpression(smooth_expr, non_smooth_expr)

        smooth_part = mixed.get_smooth_part()
        non_smooth_exprs = mixed.get_non_smooth_exprs()

        # Both parts should exist
        assert smooth_part is not None
        assert len(non_smooth_exprs) == 1

        # Sum of parts should equal original
        combined = smooth_part.forward() + non_smooth_exprs[0].forward()
        assert torch.equal(combined, mixed.forward())

    def test_multiple_smooth_and_non_smooth_terms(
        self, smooth_expr, another_smooth_expr, non_smooth_expr, another_non_smooth_expr
    ):
        """Test partitioning with multiple terms of each type."""
        mixed = SumExpression(
            smooth_expr, non_smooth_expr, another_smooth_expr, another_non_smooth_expr
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

        add_expr = SumExpression(smooth_expr, another_smooth_expr)

        expected = ExprTree(
            "SumExpression",
            smooth_expr.tree(),
            another_smooth_expr.tree(),
            is_commutative=True,
        )
        assert add_expr.tree() == expected

    # ----------------------
    # Tests for is_affine
    # ----------------------

    def test_is_affine_with_affine_operands(self):
        """Test that SumExpression is affine when all operands are affine."""
        from rlaopt.expression import expr_types

        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = expr_types.variable()(torch.tensor([4.0, 5.0, 6.0]), name="y")
        c = expr_types.constant()(torch.tensor([7.0, 8.0, 9.0]))

        # Sum of variables and constants is affine
        result = x + y + c

        assert result.is_affine() is True

    def test_is_affine_with_nonaffine_operand(self):
        """Test that SumExpression is not affine when any operand is non-affine."""
        from rlaopt.expression import expr_types

        x = expr_types.variable()(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = expr_types.variable()(torch.tensor([4.0, 5.0, 6.0]), name="y")

        # x + y**2 is not affine because y**2 is not affine
        result = x + y**2

        assert result.is_affine() is False
