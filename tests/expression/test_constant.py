import pytest
import torch

from rlaopt.expression.constant import ConstExpression


@pytest.fixture
def scalar_const():
    """Create a scalar constant expression."""
    return ConstExpression(3.14)


@pytest.fixture
def int_const():
    """Create an integer constant expression."""
    return ConstExpression(5)


@pytest.fixture
def tensor_const():
    """Create a tensor constant expression."""
    return ConstExpression(torch.tensor([1.0, 2.0, 3.0]))


class TestConstExpression:
    """Test ConstExpression concrete implementation."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_float(self, scalar_const):
        """Test initialization with float value."""
        assert torch.allclose(scalar_const.value, torch.tensor(3.14))

    def test_init_with_int(self, int_const):
        """Test initialization with integer value."""
        assert torch.equal(int_const.value, torch.tensor(5))

    def test_init_with_tensor(self, tensor_const):
        """Test initialization with tensor value."""
        assert torch.equal(tensor_const.value, torch.tensor([1.0, 2.0, 3.0]))

    def test_value_stored_as_buffer_not_parameter(self, scalar_const):
        """Test constant is stored as buffer, not parameter."""
        # Buffers don't appear in parameters
        assert len(list(scalar_const.parameters())) == 0
        # But do appear in buffers
        assert len(list(scalar_const.buffers())) == 1

    # ----------------------
    # Property tests
    # ----------------------

    def test_value_property_returns_buffer(self, scalar_const):
        """Test value property returns the stored buffer."""
        assert hasattr(scalar_const, "_value")
        assert torch.equal(scalar_const.value, scalar_const._value)

    # ----------------------
    # Abstract method implementations
    # ----------------------

    def test_is_smooth_returns_true(self, scalar_const):
        """Test constants are smooth."""
        assert scalar_const.is_smooth() is True

    def test_is_proxable_returns_true(self, scalar_const):
        """Test constants are proxable."""
        assert scalar_const.is_proxable() is True

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_value_scalar(self, scalar_const):
        """Test forward() returns the constant value for scalar."""
        result = scalar_const.forward()
        assert torch.allclose(result, torch.tensor(3.14))

    def test_forward_returns_value_tensor(self, tensor_const):
        """Test forward() returns the constant value for tensor."""
        result = tensor_const.forward()
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_forward_returns_same_reference(self, scalar_const):
        """Test forward() returns the actual buffer, not a copy."""
        result = scalar_const.forward()
        assert result is scalar_const.value

    # ----------------------
    # Operator overload tests
    # ----------------------

    def test_neg_returns_const_expression(self, scalar_const):
        """Test __neg__ returns a ConstExpression."""
        result = -scalar_const
        assert isinstance(result, ConstExpression)

    def test_neg_negates_scalar_value(self, scalar_const):
        """Test __neg__ correctly negates scalar constant."""
        result = -scalar_const
        assert torch.allclose(result.value, torch.tensor(-3.14))

    def test_neg_negates_tensor_value(self, tensor_const):
        """Test __neg__ correctly negates tensor constant."""
        result = -tensor_const
        assert torch.equal(result.value, torch.tensor([-1.0, -2.0, -3.0]))

    def test_neg_preserves_constant_type(self, scalar_const):
        """Test __neg__ result is also stored as buffer, not parameter."""
        result = -scalar_const
        assert len(list(result.parameters())) == 0
        assert len(list(result.buffers())) == 1

    def test_double_negation(self, scalar_const):
        """Test double negation returns to original value."""
        result = -(-scalar_const)
        assert torch.allclose(result.value, scalar_const.value)

    # ----------------------
    # Edge cases
    # ----------------------

    def test_zero_constant(self):
        """Test constant with zero value."""
        zero_const = ConstExpression(0)
        assert torch.equal(zero_const.value, torch.tensor(0))
        assert zero_const.forward().item() == 0

    def test_negative_constant(self):
        """Test constant with negative value."""
        neg_const = ConstExpression(-5.5)
        assert torch.allclose(neg_const.value, torch.tensor(-5.5))

    def test_multidimensional_tensor_constant(self):
        """Test constant with multidimensional tensor."""
        matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        matrix_const = ConstExpression(matrix)
        assert torch.equal(matrix_const.value, matrix)
        assert matrix_const.forward().shape == (2, 2)

    def test_empty_tensor_constant(self):
        """Test constant with empty tensor."""
        empty_const = ConstExpression(torch.tensor([]))
        assert empty_const.value.numel() == 0
        assert empty_const.forward().shape == (0,)
