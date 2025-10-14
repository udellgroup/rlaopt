"""
Comprehensive test suite for the ElasticNet atom.

Tests cover:
- Initialization with various parameter types
- Forward pass computation
- Proximal operator
- Atom properties (smooth, proxable, subsamplable)
- Buffer registration
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np

from rlaopt.expression.expression import Variable
from rlaopt.atoms.elastic_net import ElasticNet


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_variable():
    """Create a simple 1D variable."""
    return Variable(5)


@pytest.fixture
def matrix_variable():
    """Create a matrix variable."""
    return Variable(3, 4)


@pytest.fixture
def variable_with_values():
    """Create a variable with pre-set values."""
    var = Variable(5)
    var.value = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    return var


# ============================================================================
# Tests for Initialization
# ============================================================================


class TestElasticNetInitialization:
    """Test suite for ElasticNet initialization."""

    def test_basic_initialization(self, simple_variable):
        """Test basic initialization with default parameters."""
        enet = ElasticNet(simple_variable)

        assert enet.l1_scaling == 1.0
        assert enet.l2_scaling == 1.0
        # Check that variable was registered (var_name should exist)
        assert hasattr(enet, "var_name")

    def test_initialization_with_custom_scaling(self, simple_variable):
        """Test initialization with custom L1 and L2 scaling."""
        enet = ElasticNet(simple_variable, l1_scaling=0.5, l2_scaling=2.0)

        assert enet.l1_scaling == 0.5
        assert enet.l2_scaling == 2.0

    def test_initialization_with_tensor_scaling(self, simple_variable):
        """Test initialization with tensor scaling factors."""
        l1_scale = torch.tensor(0.3)
        l2_scale = torch.tensor(1.5)

        enet = ElasticNet(simple_variable, l1_scaling=l1_scale, l2_scaling=l2_scale)

        assert torch.allclose(enet.l1_scaling, l1_scale)
        assert torch.allclose(enet.l2_scaling, l2_scale)

    def test_initialization_with_zero_scaling(self, simple_variable):
        """Test initialization with zero scaling factors."""
        enet = ElasticNet(simple_variable, l1_scaling=0.0, l2_scaling=0.0)

        assert enet.l1_scaling == 0.0
        assert enet.l2_scaling == 0.0

    def test_initialization_with_matrix_variable(self, matrix_variable):
        """Test initialization with a matrix variable."""
        enet = ElasticNet(matrix_variable)

        # Check that variable was registered
        assert hasattr(enet, "var_name")

    def test_initialization_rejects_non_variable(self):
        """Test that initialization rejects non-Variable inputs."""
        with pytest.raises(TypeError, match="Expected Variable"):
            ElasticNet(torch.randn(5))

        with pytest.raises(TypeError, match="Expected Variable"):
            ElasticNet([1, 2, 3, 4, 5])

    def test_buffer_registration(self, simple_variable):
        """Test that scaling factors are registered as buffers."""
        enet = ElasticNet(simple_variable, l1_scaling=0.7, l2_scaling=1.3)

        # Check that buffers are accessible
        assert hasattr(enet, "l1_scaling")
        assert hasattr(enet, "l2_scaling")


# ============================================================================
# Tests for Forward Pass
# ============================================================================


class TestElasticNetForward:
    """Test suite for ElasticNet forward computation."""

    def test_forward_with_positive_values(self):
        """Test forward pass with positive values."""
        var = Variable(3)
        var.value = torch.tensor([1.0, 2.0, 3.0])

        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)
        result = enet.forward()

        # L1: |1| + |2| + |3| = 6
        # L2: (1^2 + 2^2 + 3^2) / 2 = 14 / 2 = 7
        # Total: 1*6 + 1*7 = 13
        expected = 13.0
        assert torch.allclose(result, torch.tensor(expected))

    def test_forward_with_negative_values(self):
        """Test forward pass with negative values."""
        var = Variable(3)
        var.value = torch.tensor([-1.0, -2.0, -3.0])

        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)
        result = enet.forward()

        # L1: |-1| + |-2| + |-3| = 6
        # L2: (1 + 4 + 9) / 2 = 7
        # Total: 13
        expected = 13.0
        assert torch.allclose(result, torch.tensor(expected))

    def test_forward_with_mixed_values(self, variable_with_values):
        """Test forward pass with mixed positive and negative values."""
        enet = ElasticNet(variable_with_values, l1_scaling=1.0, l2_scaling=1.0)
        result = enet.forward()

        # Values: [1, -2, 3, -4, 5]
        # L1: 1 + 2 + 3 + 4 + 5 = 15
        # L2: (1 + 4 + 9 + 16 + 25) / 2 = 55 / 2 = 27.5
        # Total: 15 + 27.5 = 42.5
        expected = 42.5
        assert torch.allclose(result, torch.tensor(expected))

    def test_forward_with_zeros(self):
        """Test forward pass with zero values."""
        var = Variable(4)
        var.value = torch.zeros(4)

        enet = ElasticNet(var)
        result = enet.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_with_custom_scaling(self):
        """Test forward pass with custom scaling factors."""
        var = Variable(3)
        var.value = torch.tensor([1.0, 2.0, 3.0])

        enet = ElasticNet(var, l1_scaling=0.5, l2_scaling=2.0)
        result = enet.forward()

        # L1: 6, L2: 14
        # Total: 0.5*6 + 2.0*7 = 3 + 14 = 17
        expected = 17.0
        assert torch.allclose(result, torch.tensor(expected))

    def test_forward_with_matrix(self):
        """Test forward pass with matrix variable."""
        var = Variable(2, 3)
        var.value = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])

        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)
        result = enet.forward()

        # L1: 1 + 2 + 3 + 4 + 5 + 6 = 21
        # L2: (1 + 4 + 9 + 16 + 25 + 36) / 2 = 91 / 2 = 45.5
        # Total: 21 + 45.5 = 66.5
        expected = 66.5
        assert torch.allclose(result, torch.tensor(expected))

    def test_forward_returns_scalar(self, variable_with_values):
        """Test that forward always returns a scalar."""
        enet = ElasticNet(variable_with_values)
        result = enet.forward()

        assert result.shape == torch.Size([])
        assert result.dim() == 0

    def test_forward_gradient_flow(self):
        """Test that gradients flow through forward pass."""
        var = Variable(3)
        var.value = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)
        result = enet.forward()

        # The L2 term should allow gradients, L1 won't at non-zero points
        # But we can still backprop through the computation
        result.backward()
        assert var.value.grad is not None


# ============================================================================
# Tests for Proximal Operator
# ============================================================================


class TestElasticNetProx:
    """Test suite for ElasticNet proximal operator."""

    def test_prox_basic(self):
        """Test basic proximal operator computation."""
        var = Variable(3)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([2.0, -2.0, 0.5])
        prox_scaling = 1.0

        result = enet.prox(location, prox_scaling)

        assert result.shape == location.shape
        assert not torch.isnan(result).any()

    def test_prox_soft_thresholding(self):
        """Test that prox applies soft thresholding correctly."""
        var = Variable(1)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=0.0)

        # With L2=0, this is pure soft thresholding
        location = torch.tensor([3.0])
        prox_scaling = 1.0
        threshold = 1.0  # l1_scaling * prox_scaling

        result = enet.prox(location, prox_scaling)

        # Should shrink by threshold: 3 - 1 = 2
        expected = torch.tensor([2.0])
        assert torch.allclose(result, expected)

    def test_prox_with_negative_location(self):
        """Test proximal operator with negative location."""
        var = Variable(1)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=0.0)

        location = torch.tensor([-3.0])
        prox_scaling = 1.0

        result = enet.prox(location, prox_scaling)

        # Should shrink by threshold: -3 + 1 = -2
        expected = torch.tensor([-2.0])
        assert torch.allclose(result, expected)

    def test_prox_thresholding_to_zero(self):
        """Test that small values are thresholded to zero."""
        var = Variable(3)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=0.0)

        location = torch.tensor([0.5, -0.5, 0.0])
        prox_scaling = 1.0

        result = enet.prox(location, prox_scaling)

        # All should be thresholded to zero
        expected = torch.zeros(3)
        assert torch.allclose(result, expected)

    def test_prox_with_l2_regularization(self):
        """Test proximal operator with L2 regularization."""
        var = Variable(1)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=2.0)

        location = torch.tensor([5.0])
        prox_scaling = 1.0

        result = enet.prox(location, prox_scaling)

        # threshold = 1.0 * 1.0 = 1.0
        # l2_term = 1 + 1.0 * 2.0 = 3.0
        # result = (5.0 - 1.0) / 3.0 = 4.0 / 3.0
        expected = torch.tensor([4.0 / 3.0])
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_prox_with_different_scaling(self):
        """Test proximal operator with different prox_scaling."""
        var = Variable(1)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([3.0])
        prox_scaling = 0.5

        result = enet.prox(location, prox_scaling)

        # threshold = 1.0 * 0.5 = 0.5
        # l2_term = 1 + 0.5 * 1.0 = 1.5
        # result = (3.0 - 0.5) / 1.5 = 2.5 / 1.5
        expected = torch.tensor([2.5 / 1.5])
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_prox_symmetry(self):
        """Test that prox is symmetric around zero."""
        var = Variable(1)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        prox_scaling = 1.0

        result_pos = enet.prox(torch.tensor([3.0]), prox_scaling)
        result_neg = enet.prox(torch.tensor([-3.0]), prox_scaling)

        assert torch.allclose(result_pos, -result_neg)

    def test_prox_with_matrix(self):
        """Test proximal operator with matrix input."""
        var = Variable(2, 3)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([[2.0, -2.0, 0.5], [-3.0, 4.0, 0.0]])
        prox_scaling = 1.0

        result = enet.prox(location, prox_scaling)

        assert result.shape == location.shape
        assert not torch.isnan(result).any()

    def test_prox_gradient_flow(self):
        """Test that gradients flow through prox operator."""
        var = Variable(3)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([2.0, -2.0, 3.0], requires_grad=True)
        prox_scaling = 1.0

        result = enet.prox(location, prox_scaling)
        loss = result.sum()
        loss.backward()

        assert location.grad is not None


# ============================================================================
# Tests for Atom Properties
# ============================================================================


class TestElasticNetProperties:
    """Test suite for ElasticNet atom properties."""

    def test_is_smooth(self, simple_variable):
        """Test that elastic net is not smooth."""
        enet = ElasticNet(simple_variable)
        assert enet.is_smooth() is False

    def test_is_proxable(self, simple_variable):
        """Test that elastic net is proxable."""
        enet = ElasticNet(simple_variable)
        assert enet.is_proxable() is True

    def test_is_subsamplable(self, simple_variable):
        """Test that elastic net is not subsamplable."""
        enet = ElasticNet(simple_variable)
        assert enet.is_subsamplable() is False

    def test_subsample_raises_error(self, simple_variable):
        """Test that subsample raises NotImplementedError."""
        enet = ElasticNet(simple_variable)

        with pytest.raises(NotImplementedError, match="does not support subsampling"):
            enet.subsample([0, 1, 2])

    def test_to_cvxpy_raises_error(self, simple_variable):
        """Test that to_cvxpy raises NotImplementedError."""
        enet = ElasticNet(simple_variable)

        with pytest.raises(NotImplementedError, match="not supported"):
            enet.to_cvxpy()


# ============================================================================
# Integration Tests
# ============================================================================


class TestElasticNetIntegration:
    """Integration tests for ElasticNet atom."""

    def test_forward_and_prox_consistency(self):
        """Test consistency between forward and prox operations."""
        var = Variable(5)
        var.value = torch.randn(5)

        enet = ElasticNet(var, l1_scaling=0.5, l2_scaling=1.0)

        # Forward pass
        forward_result = enet.forward()
        assert forward_result >= 0  # Penalty should be non-negative

        # Prox operation
        prox_result = enet.prox(var.value, prox_scaling=1.0)
        assert prox_result.shape == var.value.shape

    def test_multiple_forward_calls(self, variable_with_values):
        """Test that multiple forward calls give consistent results."""
        enet = ElasticNet(variable_with_values)

        result1 = enet.forward()
        result2 = enet.forward()

        assert torch.allclose(result1, result2)

    def test_prox_reduces_penalty(self):
        """Test that prox operator reduces the penalty value."""
        var = Variable(5)
        initial_value = torch.randn(5) * 5  # Large values
        var.value = initial_value.clone()

        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        # Original penalty
        original_penalty = enet.forward()

        # Apply prox and compute new penalty
        prox_result = enet.prox(initial_value, prox_scaling=1.0)
        var.value = prox_result
        new_penalty = enet.forward()

        # Prox should reduce the penalty
        assert new_penalty <= original_penalty

    def test_edge_case_all_zeros(self):
        """Test edge case with all zero values."""
        var = Variable(10)
        var.value = torch.zeros(10)

        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        forward_result = enet.forward()
        assert torch.allclose(forward_result, torch.tensor(0.0))

        prox_result = enet.prox(torch.zeros(10), prox_scaling=1.0)
        assert torch.allclose(prox_result, torch.zeros(10))

    def test_scaling_effects(self):
        """Test that scaling factors have expected effects."""
        var = Variable(3)
        var.value = torch.tensor([1.0, 2.0, 3.0])

        # Baseline
        enet1 = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)
        result1 = enet1.forward()

        # Double L1
        enet2 = ElasticNet(var, l1_scaling=2.0, l2_scaling=1.0)
        result2 = enet2.forward()

        # Double L2
        enet3 = ElasticNet(var, l1_scaling=1.0, l2_scaling=2.0)
        result3 = enet3.forward()

        # All should be different
        assert result2 > result1
        assert result3 > result1
        assert not torch.allclose(result2, result3)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestElasticNetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_values(self):
        """Test with very large values."""
        var = Variable(3)
        var.value = torch.tensor([1e6, -1e6, 1e6])

        enet = ElasticNet(var)
        result = enet.forward()

        assert not torch.isnan(result)
        assert not torch.isinf(result)
        assert result > 0

    def test_very_small_values(self):
        """Test with very small values."""
        var = Variable(3)
        var.value = torch.tensor([1e-10, -1e-10, 1e-10])

        enet = ElasticNet(var)
        result = enet.forward()

        assert result >= 0
        # For very small values, result should also be very small
        assert result < 1e-9  # Relaxed threshold since penalty scales with input

    def test_nan_handling(self):
        """Test behavior with NaN values."""
        var = Variable(3)
        var.value = torch.tensor([1.0, float("nan"), 3.0])

        enet = ElasticNet(var)
        result = enet.forward()

        # Result should be NaN if input contains NaN
        assert torch.isnan(result)

    def test_large_prox_scaling(self):
        """Test prox with very large scaling."""
        var = Variable(3)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([1.0, 2.0, 3.0])
        prox_scaling = 1000.0

        result = enet.prox(location, prox_scaling)

        # With very large scaling, result should be close to zero
        assert torch.allclose(result, torch.zeros(3), atol=0.01)

    def test_zero_prox_scaling(self):
        """Test prox with zero scaling."""
        var = Variable(3)
        enet = ElasticNet(var, l1_scaling=1.0, l2_scaling=1.0)

        location = torch.tensor([1.0, 2.0, 3.0])
        prox_scaling = 0.0

        result = enet.prox(location, prox_scaling)

        # With zero scaling, should return input (no shrinkage)
        assert torch.allclose(result, location)

    @pytest.mark.parametrize("size", [1, 10, 100, 1000])
    def test_various_sizes(self, size):
        """Test with various variable sizes."""
        var = Variable(size)
        var.value = torch.randn(size)

        enet = ElasticNet(var)
        result = enet.forward()

        assert result.shape == torch.Size([])
        assert result >= 0
