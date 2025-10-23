import pytest
import torch

from rlaopt.atoms.non_negative import NonNegative
from rlaopt.expression.variable import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="x")


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    return Variable((3, 4), name="X")


class TestNonNegative:
    """Test NonNegative constraint atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_sets_lower_bound_to_zero(self, vector_var):
        """Test initialization sets lower bound to 0."""
        nonneg = NonNegative(vector_var)
        assert torch.allclose(nonneg.lower, torch.tensor(0.0))

    def test_init_sets_upper_bound_to_infinity(self, vector_var):
        """Test initialization sets upper bound to infinity."""
        nonneg = NonNegative(vector_var)
        assert torch.isinf(nonneg.upper) and nonneg.upper > 0

    def test_init_with_matrix_variable(self, matrix_var):
        """Test initialization works with matrix variables."""
        nonneg = NonNegative(matrix_var)
        assert nonneg.lower is not None
        assert nonneg.upper is not None

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_zero_for_nonnegative_values(self, vector_var):
        """Test forward() returns 0 when all values are non-negative."""
        nonneg = NonNegative(vector_var)
        vector_var.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nonneg.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_zero_for_zero_values(self, vector_var):
        """Test forward() returns 0 when values are exactly zero."""
        nonneg = NonNegative(vector_var)
        vector_var.value.data = torch.zeros(5)
        result = nonneg.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_inf_for_negative_values(self, vector_var):
        """Test forward() returns infinity when any value is negative."""
        nonneg = NonNegative(vector_var)
        vector_var.value.data = torch.tensor([1.0, -0.1, 3.0, 4.0, 5.0])
        result = nonneg.forward()
        assert torch.isinf(result)

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_false(self, vector_var):
        """Test non-negativity constraint is not smooth."""
        nonneg = NonNegative(vector_var)
        assert nonneg.is_smooth() is False

    def test_is_proxable_returns_true(self, vector_var):
        """Test non-negativity constraint is proxable."""
        nonneg = NonNegative(vector_var)
        assert nonneg.is_proxable() is True

    # ----------------------
    # Proximal operator tests
    # ----------------------

    def test_prox_clamps_negative_values_to_zero(self, vector_var):
        """Test prox() clamps negative values to zero."""
        nonneg = NonNegative(vector_var)
        location = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        result = nonneg.prox(location, prox_scaling=1.0)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_prox_preserves_positive_values(self, vector_var):
        """Test prox() leaves positive values unchanged."""
        nonneg = NonNegative(vector_var)
        location = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        result = nonneg.prox(location, prox_scaling=1.0)
        assert torch.allclose(result, location)

    def test_prox_is_independent_of_scaling(self, vector_var):
        """Test prox() gives same result regardless of scaling parameter."""
        nonneg = NonNegative(vector_var)
        location = torch.tensor([-1.0, 0.5, 2.0, -3.0, 1.5])

        result1 = nonneg.prox(location, prox_scaling=1.0)
        result2 = nonneg.prox(location, prox_scaling=100.0)
        assert torch.allclose(result1, result2)

    def test_prox_with_all_negative_values(self, vector_var):
        """Test prox() with all negative values returns zeros."""
        nonneg = NonNegative(vector_var)
        location = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0])

        result = nonneg.prox(location, prox_scaling=1.0)
        assert torch.allclose(result, torch.zeros(5))

    def test_prox_with_matrix_input(self, matrix_var):
        """Test prox() works with matrix variables."""
        nonneg = NonNegative(matrix_var)
        location = torch.randn(3, 4)  # Mix of positive and negative

        result = nonneg.prox(location, prox_scaling=1.0)

        # All values should be non-negative
        assert torch.all(result >= 0)
        # Positive values should be unchanged
        assert torch.allclose(result[location >= 0], location[location >= 0])

    # ----------------------
    # Edge cases
    # ----------------------

    def test_prox_at_boundary(self, vector_var):
        """Test prox() of zeros returns zeros."""
        nonneg = NonNegative(vector_var)
        location = torch.zeros(5)

        result = nonneg.prox(location, prox_scaling=1.0)
        assert torch.allclose(result, torch.zeros(5))

    def test_with_large_positive_values(self, vector_var):
        """Test constraint allows arbitrarily large positive values."""
        nonneg = NonNegative(vector_var)
        vector_var.value.data = torch.tensor([1e6, 1e9, 1e12, 1e15, 1e18])
        result = nonneg.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_gradient_flows_through_prox(self, vector_var):
        """Test gradients flow through prox operator."""
        nonneg = NonNegative(vector_var)
        location = torch.tensor([-1.0, 0.5, 2.0, -0.5, 3.0], requires_grad=True)

        result = nonneg.prox(location, prox_scaling=1.0)
        loss = result.sum()
        loss.backward()

        assert location.grad is not None
        # Gradient should be zero for negative inputs (clamped to 0)
        # and 1 for positive inputs (passed through)
        expected_grad = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
        assert torch.allclose(location.grad, expected_grad)
