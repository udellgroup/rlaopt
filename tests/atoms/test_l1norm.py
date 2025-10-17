"""Comprehensive tests for the L1Norm atom."""

import pytest
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.atoms.l1norm import L1Norm
from rlaopt.expression import Variable

# ===============================
# Test Initialization
# ===============================


class TestL1NormInit:
    """Tests for L1Norm initialization."""

    def test_init_with_variable(self):
        """Test basic initialization with a Variable."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)

        assert isinstance(l1, L1Norm)
        assert isinstance(l1, AtomExpression)
        assert l1.var_name == "x"

    def test_init_with_custom_scaling(self):
        """Test initialization with custom scaling factor."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=2.5)

        assert hasattr(l1, "scaling")
        assert torch.allclose(l1.scaling, torch.tensor(2.5))

    def test_init_with_default_scaling(self):
        """Test that default scaling is 1.0."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)

        assert torch.allclose(l1.scaling, torch.tensor(1.0))

    def test_init_with_zero_scaling(self):
        """Test initialization with zero scaling."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=0.0)

        assert torch.allclose(l1.scaling, torch.tensor(0.0))

    def test_init_with_negative_scaling(self):
        """Test initialization with negative scaling."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=-1.5)

        assert torch.allclose(l1.scaling, torch.tensor(-1.5))

    def test_init_with_tensor_scaling(self):
        """Test initialization with tensor scaling."""
        x = Variable((5,), name="x")
        scaling = torch.tensor(3.0)
        l1 = L1Norm(x, scaling=scaling)

        assert torch.allclose(l1.scaling, torch.tensor(3.0))

    def test_init_with_non_variable_raises_error(self):
        """Test that non-Variable input raises TypeError."""
        with pytest.raises(TypeError, match="Expected Variable"):
            L1Norm(torch.ones(5))

    def test_init_with_expression_raises_error(self):
        """Test that Expression input raises TypeError."""
        x = Variable((5,), name="x")
        y = Variable((5,), name="y")
        expr = x + y

        with pytest.raises(TypeError, match="Expected Variable"):
            L1Norm(expr)

    def test_init_with_none_raises_error(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected Variable"):
            L1Norm(None)

    def test_init_with_different_shapes(self):
        """Test initialization with different variable shapes."""
        shapes = [(5,), (10,), (3, 4), (2, 3, 4)]

        for shape in shapes:
            x = Variable(shape, name="x")
            l1 = L1Norm(x)
            assert l1.var_name == "x"


# ===============================
# Test Properties
# ===============================


class TestL1NormProperties:
    """Tests for L1Norm mathematical properties."""

    def test_is_smooth(self):
        """Test that L1Norm is not smooth."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)

        assert l1.is_smooth() is False

    def test_is_proxable(self):
        """Test that L1Norm is proxable."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)

        assert l1.is_proxable() is True

    def test_is_subsamplable(self):
        """Test that L1Norm is not subsamplable."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)

        assert l1.is_subsamplable() is False

    def test_subsample_raises_error(self):
        """Test that subsample raises NotImplementedError."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)
        indices = torch.tensor([0, 1, 2])

        with pytest.raises(NotImplementedError, match="cannot be subsampled"):
            l1.subsample(indices)


# ===============================
# Test Forward Pass
# ===============================


class TestL1NormForward:
    """Tests for L1Norm forward evaluation."""

    def test_forward_with_positive_values(self):
        """Test forward pass with positive values."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(15.0)  # 1 + 2 + 3 + 4 + 5

        assert torch.allclose(result, expected)

    def test_forward_with_negative_values(self):
        """Test forward pass with negative values."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(15.0)  # |-1| + |-2| + |-3| + |-4| + |-5|

        assert torch.allclose(result, expected)

    def test_forward_with_mixed_values(self):
        """Test forward pass with mixed positive and negative values."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(15.0)  # 1 + 2 + 3 + 4 + 5

        assert torch.allclose(result, expected)

    def test_forward_with_zeros(self):
        """Test forward pass with zero values."""
        x = Variable((5,), name="x")
        x.value.data = torch.zeros(5)
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(0.0)

        assert torch.allclose(result, expected)

    def test_forward_with_scaling(self):
        """Test forward pass with custom scaling."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5)
        l1 = L1Norm(x, scaling=2.5)

        result = l1.forward()
        expected = torch.tensor(12.5)  # 2.5 * (1 + 1 + 1 + 1 + 1)

        assert torch.allclose(result, expected)

    def test_forward_with_zero_scaling(self):
        """Test forward pass with zero scaling."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        l1 = L1Norm(x, scaling=0.0)

        result = l1.forward()
        expected = torch.tensor(0.0)

        assert torch.allclose(result, expected)

    def test_forward_with_negative_scaling(self):
        """Test forward pass with negative scaling."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5)
        l1 = L1Norm(x, scaling=-2.0)

        result = l1.forward()
        expected = torch.tensor(-10.0)  # -2.0 * 5

        assert torch.allclose(result, expected)

    def test_forward_multidimensional(self):
        """Test forward pass with multidimensional variable."""
        x = Variable((3, 4), name="x")
        x.value.data = torch.ones(3, 4)
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(12.0)  # 3 * 4 = 12 ones

        assert torch.allclose(result, expected)

    def test_forward_returns_scalar(self):
        """Test that forward returns a scalar tensor."""
        x = Variable((5,), name="x")
        x.value.data = torch.randn(5)
        l1 = L1Norm(x)

        result = l1.forward()

        assert result.dim() == 0  # Scalar tensor
        assert result.shape == torch.Size([])

    def test_forward_with_small_values(self):
        """Test forward pass with very small values."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(15e-10)

        assert torch.allclose(result, expected, atol=1e-15)

    def test_forward_with_large_values(self):
        """Test forward pass with large values."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1e6, 2e6, 3e6])
        l1 = L1Norm(x)

        result = l1.forward()
        expected = torch.tensor(6e6)

        assert torch.allclose(result, expected)


# ===============================
# Test Proximal Operator
# ===============================


class TestL1NormProx:
    """Tests for L1Norm proximal operator (soft-thresholding)."""

    def test_prox_soft_thresholding_positive(self):
        """Test soft-thresholding with positive values."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0])
        prox_scaling = 1.0

        result = l1.prox(location, prox_scaling)
        expected = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])

        assert torch.allclose(result, expected)

    def test_prox_soft_thresholding_negative(self):
        """Test soft-thresholding with negative values."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0])
        prox_scaling = 1.0

        result = l1.prox(location, prox_scaling)
        expected = torch.tensor([-2.0, -1.0, 0.0, 0.0, 0.0])

        assert torch.allclose(result, expected)

    def test_prox_soft_thresholding_mixed(self):
        """Test soft-thresholding with mixed positive and negative values."""
        x = Variable((6,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([3.0, -2.0, 1.5, -1.5, 0.5, -0.5])
        prox_scaling = 1.0

        result = l1.prox(location, prox_scaling)
        expected = torch.tensor([2.0, -1.0, 0.5, -0.5, 0.0, 0.0])

        assert torch.allclose(result, expected)

    def test_prox_with_different_threshold(self):
        """Test soft-thresholding with different threshold."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        prox_scaling = 2.0

        result = l1.prox(location, prox_scaling)
        expected = torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0])

        assert torch.allclose(result, expected)

    def test_prox_with_scaling_factor(self):
        """Test soft-thresholding with L1 scaling factor."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=2.0)

        location = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        prox_scaling = 1.0

        result = l1.prox(location, prox_scaling)
        # threshold = 2.0 * 1.0 = 2.0
        expected = torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0])

        assert torch.allclose(result, expected)

    def test_prox_with_zero_prox_scaling(self):
        """Test soft-thresholding with zero prox_scaling (no thresholding)."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([3.0, -2.0, 1.0, -1.0, 0.0])
        prox_scaling = 0.0

        result = l1.prox(location, prox_scaling)
        # No thresholding, should return location unchanged

        assert torch.allclose(result, location)

    def test_prox_with_large_threshold(self):
        """Test soft-thresholding with large threshold (all zeros)."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        prox_scaling = 10.0

        result = l1.prox(location, prox_scaling)
        expected = torch.zeros(5)

        assert torch.allclose(result, expected)

    def test_prox_multidimensional(self):
        """Test soft-thresholding with multidimensional input."""
        x = Variable((3, 4), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor(
            [[3.0, 2.0, 1.0, 0.5], [-3.0, -2.0, -1.0, -0.5], [1.5, -1.5, 0.5, -0.5]]
        )
        prox_scaling = 1.0

        result = l1.prox(location, prox_scaling)
        expected = torch.tensor(
            [[2.0, 1.0, 0.0, 0.0], [-2.0, -1.0, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0]]
        )

        assert torch.allclose(result, expected)

    def test_prox_preserves_shape(self):
        """Test that prox preserves input shape."""
        x = Variable((3, 4), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.randn(3, 4)
        result = l1.prox(location, 1.0)

        assert result.shape == location.shape

    def test_prox_exact_at_threshold(self):
        """Test soft-thresholding exactly at threshold."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.0])
        prox_scaling = 1.0

        result = l1.prox(location, prox_scaling)
        expected = torch.tensor([0.0, 0.0, 1.0, -1.0, 0.0])

        assert torch.allclose(result, expected)

    def test_prox_mathematical_property(self):
        """Test that prox satisfies mathematical properties."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        location = torch.tensor([3.0, -2.0, 1.0, -1.0, 0.0])
        prox_scaling = 0.5

        result = l1.prox(location, prox_scaling)

        # prox should shrink magnitude
        assert torch.all(torch.abs(result) <= torch.abs(location))

        # prox should preserve sign
        assert torch.all(torch.sign(result) == torch.sign(location))


# ===============================
# Test Gradient Flow
# ===============================


class TestL1NormGradients:
    """Tests for gradient computation through L1Norm."""

    def test_gradient_flow_positive_values(self):
        """Test gradient flow with positive values."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        l1 = L1Norm(x)

        result = l1.forward()
        result.backward()

        # Subgradient of |x| is sign(x)
        expected_grad = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(x.value.grad, expected_grad)

    def test_gradient_flow_negative_values(self):
        """Test gradient flow with negative values."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        l1 = L1Norm(x)

        result = l1.forward()
        result.backward()

        # Subgradient of |-x| is -sign(x) = sign(-x)
        expected_grad = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0])
        assert torch.allclose(x.value.grad, expected_grad)

    def test_gradient_flow_with_scaling(self):
        """Test gradient flow with scaling factor."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5)
        l1 = L1Norm(x, scaling=2.5)

        result = l1.forward()
        result.backward()

        # Gradient should be scaled
        expected_grad = torch.ones(5) * 2.5
        assert torch.allclose(x.value.grad, expected_grad)

    def test_gradient_at_zero(self):
        """Test gradient computation at zero (subgradient)."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 0.0, -1.0])
        l1 = L1Norm(x)

        result = l1.forward()
        result.backward()

        # At x=0, subgradient can be anything in [-1, 1]
        # PyTorch's abs gradient at 0 is 0
        assert x.value.grad[1] == 0.0
        assert x.value.grad[0] == 1.0
        assert x.value.grad[2] == -1.0


# ===============================
# Test Integration & Special Cases
# ===============================


class TestL1NormIntegration:
    """Integration tests and special cases for L1Norm."""

    def test_l1_norm_as_regularizer(self):
        """Test L1Norm used as a regularizer in optimization."""
        x = Variable((10,), name="x")
        x.value.data = torch.randn(10)

        # Create L1 regularizer
        reg = L1Norm(x, scaling=0.1)

        # Compute regularization term
        reg_value = reg.forward()

        # Should be positive
        assert reg_value >= 0

        # Should equal scaled sum of absolute values
        expected = 0.1 * torch.sum(torch.abs(x.value))
        assert torch.allclose(reg_value, expected)

    def test_l1_norm_sparsity_inducing(self):
        """Test that L1 prox induces sparsity."""
        x = Variable((10,), name="x")
        l1 = L1Norm(x, scaling=1.0)

        # Small random values
        location = torch.randn(10) * 0.3
        result = l1.prox(location, prox_scaling=0.5)

        # Many values should be exactly zero
        num_zeros = torch.sum(result == 0.0).item()
        assert num_zeros > 0

    def test_l1_norm_state_dict(self):
        """Test state dict save and load."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 3
        l1 = L1Norm(x, scaling=2.5)

        # Save state
        state = l1.state_dict()

        # Check contents
        assert "x" in state
        assert "scaling" in state
        assert torch.allclose(state["scaling"], torch.tensor(2.5))

    def test_l1_norm_parameter_tracking(self):
        """Test that L1Norm tracks variable parameters."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x)

        params = list(l1.parameters())
        assert len(params) == 1
        assert params[0].shape == torch.Size([5])

    def test_l1_norm_buffer_tracking(self):
        """Test that L1Norm tracks scaling as buffer."""
        x = Variable((5,), name="x")
        l1 = L1Norm(x, scaling=3.0)

        buffers = dict(l1.named_buffers())
        assert "scaling" in buffers
        assert torch.allclose(buffers["scaling"], torch.tensor(3.0))

    def test_l1_norm_device_transfer(self):
        """Test moving L1Norm to device."""
        x = Variable((5,), name="x", device=torch.device("cpu"))
        l1 = L1Norm(x, scaling=2.0)

        l1 = l1.to("cpu")

        assert l1.get_variable("x").device.type == "cpu"
        assert l1.scaling.device.type == "cpu"

    def test_l1_norm_dtype_conversion(self):
        """Test L1Norm dtype conversion."""
        x = Variable((5,), name="x", dtype=torch.float32)
        l1 = L1Norm(x)

        l1 = l1.to(torch.float64)

        assert l1.get_variable("x").dtype == torch.float64

    def test_l1_norm_in_optimization_loop(self):
        """Test L1Norm in a simple optimization loop."""
        x = Variable((5,), name="x")
        x.value.data = torch.randn(5)
        l1 = L1Norm(x, scaling=0.1)

        optimizer = torch.optim.SGD([x.value], lr=0.01)

        initial_norm = l1.forward().item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = l1.forward()
            loss.backward()
            optimizer.step()

        final_norm = l1.forward().item()

        # L1 norm should decrease
        assert final_norm < initial_norm

    def test_l1_norm_composition_with_other_atoms(self):
        """Test L1Norm can be composed with other operations."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5) * 2

        l1 = L1Norm(x, scaling=0.5)
        reg_term = l1.forward()

        # Can be added to other losses
        data_loss = torch.sum(x.value**2)
        total_loss = data_loss + reg_term

        assert total_loss.requires_grad
        total_loss.backward()
        assert x.value.grad is not None


# ===============================
# Test Edge Cases
# ===============================


class TestL1NormEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_variable(self):
        """Test L1Norm with empty variable."""
        x = Variable((0,), name="x")
        l1 = L1Norm(x)

        result = l1.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_single_element_variable(self):
        """Test L1Norm with single element."""
        x = Variable((1,), name="x")
        x.value.data = torch.tensor([5.0])
        l1 = L1Norm(x)

        result = l1.forward()
        assert torch.allclose(result, torch.tensor(5.0))

    def test_very_large_scaling(self):
        """Test L1Norm with very large scaling."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5)
        l1 = L1Norm(x, scaling=1e6)

        result = l1.forward()
        assert torch.allclose(result, torch.tensor(5e6))

    def test_very_small_scaling(self):
        """Test L1Norm with very small scaling."""
        x = Variable((5,), name="x")
        x.value.data = torch.ones(5)
        l1 = L1Norm(x, scaling=1e-10)

        result = l1.forward()
        assert torch.allclose(result, torch.tensor(5e-10), atol=1e-15)

    def test_inf_values_handling(self):
        """Test L1Norm with inf values."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, float("inf"), 2.0])
        l1 = L1Norm(x)

        result = l1.forward()
        assert torch.isinf(result)

    def test_nan_values_handling(self):
        """Test L1Norm with NaN values."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, float("nan"), 2.0])
        l1 = L1Norm(x)

        result = l1.forward()
        assert torch.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
