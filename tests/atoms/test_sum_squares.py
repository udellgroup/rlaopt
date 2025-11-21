"""Tests for SumSquares atom."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms.sum_squares import SumSquares
from rlaopt.expression import Expression, Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    x = Variable(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
    return x


@pytest.fixture
def simple_affine(vector_var):
    """Create a simple affine transformation."""
    b = torch.zeros(5)
    return vector_var + b


@pytest.fixture
def nontrivial_affine(vector_var):
    """Create a non-trivial affine transformation."""
    A = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    b = torch.tensor([1.0, -1.0, 0.5])
    return A @ vector_var + b


class TestSumSquares:
    """Test SumSquares atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_variable(self, vector_var):
        """Test initialization with Variable input."""
        ss = SumSquares(vector_var)
        assert isinstance(ss.get_input("x"), Variable)

    def test_init_with_expression(self, simple_affine):
        """Test initialization with Expression input."""
        ss = SumSquares(simple_affine)
        assert isinstance(ss.get_input("x"), Expression)

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_with_variable(self, vector_var):
        """Test forward() computes sum of squares for Variable."""
        ss = SumSquares(vector_var)
        result = ss.forward()
        # [1,2,3,4,5]^2 = 1+4+9+16+25 = 55
        expected = torch.tensor(55.0)
        assert torch.allclose(result, expected)

    def test_forward_with_zero_variable(self):
        """Test forward() with zero variable."""
        x = Variable((5,), name="x")
        x.value.data = torch.zeros(5)
        ss = SumSquares(x)
        result = ss.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_with_affine_expression(self, simple_affine):
        """Test forward() with affine expression."""
        ss = SumSquares(simple_affine)
        result = ss.forward()
        # Affine is identity, so same as variable: 55
        expected = torch.tensor(55.0)
        assert torch.allclose(result, expected)

    def test_forward_with_nontrivial_affine(self, nontrivial_affine):
        """Test forward() with non-trivial affine transformation."""
        ss = SumSquares(nontrivial_affine)
        result = ss.forward()

        # A @ x + b where x = [1,2,3,4,5]
        # Row 0: 1*1 + 1 = 2
        # Row 1: 2*2 + (-1) = 3
        # Row 2: 1*3 + 0.5 = 3.5
        # Sum of squares: 4 + 9 + 12.25 = 25.25
        expected = torch.tensor(25.25)
        assert torch.allclose(result, expected)

    def test_forward_with_general_expression(self, vector_var):
        """Test forward() with general expression (addition)."""
        expr = vector_var + 1.0
        ss = SumSquares(expr)
        result = ss.forward()
        # [1,2,3,4,5] + 1 = [2,3,4,5,6]
        # Sum of squares: 4+9+16+25+36 = 90
        expected = torch.tensor(90.0)
        assert torch.allclose(result, expected)

    # ----------------------
    # Smoothness tests
    # ----------------------

    def test_is_smooth_with_variable(self, vector_var):
        """Test is_smooth() returns True for Variable input."""
        ss = SumSquares(vector_var)
        assert ss.is_smooth() is True

    def test_is_smooth_with_smooth_expression(self, simple_affine):
        """Test is_smooth() returns True for smooth expression."""
        ss = SumSquares(simple_affine)
        assert ss.is_smooth() is True

    def test_is_smooth_with_nonsmooth_expression(self):
        """Test is_smooth() returns False for non-smooth expression."""

        # Create a mock non-smooth expression
        class NonSmoothExpr(Expression):
            def __init__(self):
                super().__init__()

            def is_smooth(self):
                return False

            def is_proxable(self):
                return False

            def forward(self):
                return torch.ones(5)

            def tree(self):
                raise NotImplementedError()

        nonsmooth = NonSmoothExpr()
        ss = SumSquares(nonsmooth)
        assert ss.is_smooth() is False

    # ----------------------
    # Proxability tests
    # ----------------------

    def test_is_proxable_with_variable(self, vector_var):
        """Test is_proxable() returns True for Variable input."""
        ss = SumSquares(vector_var)
        assert ss.is_proxable() is True

    def test_is_proxable_with_general_expression(self, vector_var):
        """Test is_proxable() returns False for general expression."""
        expr = vector_var + 1.0
        ss = SumSquares(expr)
        assert ss.is_proxable() is False

    # ----------------------
    # Proximal operator tests - Variable
    # ----------------------

    def test_prox_with_variable(self, vector_var):
        """Test prox() with Variable input uses scalar shrinkage."""
        ss = SumSquares(vector_var)
        location = TensorDict(
            {vector_var.name: torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])}
        )
        prox_scaling = 1.0

        result = ss.prox(location, prox_scaling)
        # Formula: location / (1 + 2*prox_scaling) = location / 3
        expected = TensorDict(
            {vector_var.name: torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]) / 3.0}
        )
        assert_allclose_td(result, expected)

    def test_prox_with_variable_different_scaling(self, vector_var):
        """Test prox() with Variable and different scaling factors."""
        ss = SumSquares(vector_var)
        location_tensor = torch.ones(5) * 6
        location = TensorDict({vector_var.name: location_tensor})

        result1 = ss.prox(location, prox_scaling=0.5)
        # 1 + 2*0.5 = 2
        expected1 = TensorDict({vector_var.name: location_tensor / 2.0})
        assert_allclose_td(result1, expected1)

        result2 = ss.prox(location, prox_scaling=2.0)
        expected2 = TensorDict({vector_var.name: location_tensor / 5.0})  # 1 + 2*2 = 5
        assert_allclose_td(result2, expected2)

    # ----------------------
    # Error handling tests
    # ----------------------

    def test_prox_raises_error_for_general_expression(self, vector_var):
        """Test prox() raises error for general expression."""
        expr = vector_var + 1.0
        ss = SumSquares(expr)

        with pytest.raises(
            NotImplementedError, match="Proximal operator .* not implemented"
        ):
            ss.prox(TensorDict({vector_var.name: torch.ones(5)}), 1.0)

    # ----------------------
    # Edge cases
    # ----------------------

    def test_prox_with_zero_location(self, vector_var):
        """Test prox() with zero location."""
        ss = SumSquares(vector_var)
        location = TensorDict({vector_var.name: torch.zeros(5)})
        result = ss.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.zeros(5)})
        assert_allclose_td(result, expected)
