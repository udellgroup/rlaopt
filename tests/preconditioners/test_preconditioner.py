import pytest
import torch
from rlaopt.preconditioners import Preconditioner
from rlaopt.preconditioners.configs import PreconditionerConfig


class TestPreconditionerBase:
    """Tests for the Preconditioner abstract base class."""

    @pytest.fixture
    def mock_preconditioner(self):
        """Create a concrete implementation of the Preconditioner for testing."""
        # Since Preconditioner is abstract, we need a concrete subclass to test it
        class ConcretePreconditioner(Preconditioner):
            def _update(self, A, device, *args, **kwargs):
                # Implementation for testing
                self.was_updated = True

            def _matmul(self, x):
                # Return identity operation for testing
                return x

            def _inverse_matmul_1d(self, x):
                # Track that this was called
                self.inverse_1d_called = True
                return x

            def _inverse_matmul_2d(self, x):
                # Track that this was called
                self.inverse_2d_called = True
                return x

        config = PreconditionerConfig()
        return ConcretePreconditioner(config)

    def test_matmul_input_validation(self, mock_preconditioner):
        """Test that __matmul__ validates inputs correctly."""
        # Valid 1D input
        x_1d = torch.randn(10)
        result = mock_preconditioner @ x_1d
        assert torch.equal(result, x_1d)  # Should pass through for our mock

        # Valid 2D input
        x_2d = torch.randn(10, 5)
        result = mock_preconditioner @ x_2d
        assert torch.equal(result, x_2d)  # Should pass through for our mock

        # Invalid 3D input
        x_3d = torch.randn(10, 5, 3)
        with pytest.raises(ValueError):
            mock_preconditioner @ x_3d

        # Invalid input type
        with pytest.raises((TypeError, ValueError)):
            mock_preconditioner @ "invalid"

    def test_inverse_matmul_delegation(self, mock_preconditioner):
        """Test that _inverse_matmul delegates to appropriate methods
        based on input dimensions."""
        # 1D case
        x_1d = torch.randn(10)
        mock_preconditioner.inverse_1d_called = False
        mock_preconditioner._inverse_matmul(x_1d)
        assert mock_preconditioner.inverse_1d_called

        # 2D case
        x_2d = torch.randn(10, 5)
        mock_preconditioner.inverse_2d_called = False
        mock_preconditioner._inverse_matmul(x_2d)
        assert mock_preconditioner.inverse_2d_called

        # Invalid input dimensions
        x_3d = torch.randn(10, 5, 3)
        with pytest.raises(ValueError):
            mock_preconditioner._inverse_matmul(x_3d)

    def test_inv_property(self, mock_preconditioner):
        """Test that _inv property returns an _InvPreconditioner instance."""
        inv = mock_preconditioner._inv
        assert hasattr(inv, "preconditioner")
        assert inv.preconditioner is mock_preconditioner

        # Test the __matmul__ of the inverse with 1D input
        x_1d = torch.randn(10)
        mock_preconditioner.inverse_1d_called = False
        inv @ x_1d
        assert mock_preconditioner.inverse_1d_called

        # Test with 2D input
        x_2d = torch.randn(10, 5)
        mock_preconditioner.inverse_2d_called = False
        inv @ x_2d
        assert mock_preconditioner.inverse_2d_called

    def test_inverse_matmul_compose(self, mock_preconditioner):
        """Test the _inverse_matmul_compose method."""
        # Create a test function
        def test_fn(x):
            return x * 2

        # Compose it with inverse_matmul
        composed_fn = mock_preconditioner._inverse_matmul_compose(test_fn)

        # Test with an input
        x = torch.randn(10)
        mock_preconditioner.inverse_1d_called = False
        result = composed_fn(x)

        # Check that _inverse_matmul_1d was called
        assert mock_preconditioner.inverse_1d_called

        # Since our implementation returns the input, the result should be x*2
        assert torch.equal(result, x * 2)

    def test_update_damping(self, mock_preconditioner):
        """Test the _update_damping method
        (should be a no-op for most preconditioners)."""
        # By default this should be a no-op, so just verify it can be called
        mock_preconditioner._update_damping(0.1)
        # No assertion needed since it's a no-op, just checking it doesn't error
