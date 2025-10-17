"""Comprehensive tests for the Halfspace constraint atom."""

import math

import pytest
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.atoms.halfspace import Halfspace
from rlaopt.atoms.polyhedron import Polyhedron
from rlaopt.expression import Variable

# ===============================
# Test Initialization
# ===============================


class TestHalfspaceInit:
    """Tests for Halfspace initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = 1.0

        halfspace = Halfspace(x, c=c, upper=upper)

        assert isinstance(halfspace, Halfspace)
        assert isinstance(halfspace, Polyhedron)
        assert isinstance(halfspace, AtomExpression)

    def test_init_stores_c_and_upper(self):
        """Test that c and upper are stored correctly."""
        x = Variable((5,), name="x")
        c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        upper = 10.0

        halfspace = Halfspace(x, c=c, upper=upper)

        assert torch.equal(halfspace.C, c)
        assert torch.allclose(halfspace.upper, torch.tensor(10.0))

    def test_init_with_zero_vector(self):
        """Test initialization with zero normal vector."""
        x = Variable((5,), name="x")
        c = torch.zeros(5)
        upper = 0.0

        # Should initialize but projection will be undefined
        halfspace = Halfspace(x, c=c, upper=upper)
        assert torch.equal(halfspace.C, c)

    def test_init_with_negative_upper(self):
        """Test initialization with negative upper bound."""
        x = Variable((5,), name="x")
        c = torch.ones(5)
        upper = -5.0

        halfspace = Halfspace(x, c=c, upper=upper)
        assert torch.allclose(halfspace.upper, torch.tensor(-5.0))

    def test_init_different_variable_shapes(self):
        """Test initialization with different variable shapes."""
        shapes = [(5,), (10,), (3, 4), (2, 3, 4)]

        for shape in shapes:
            x = Variable(shape, name="x")
            c = torch.randn(shape)
            upper = 1.0
            halfspace = Halfspace(x, c=c, upper=upper)
            assert halfspace.var_name == "x"

    def test_init_with_int_upper(self):
        """Test initialization with int upper bound."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = 5  # int, not float

        halfspace = Halfspace(x, c=c, upper=upper)
        assert isinstance(halfspace.upper, torch.Tensor)


# ===============================
# Test Properties
# ===============================


class TestHalfspaceProperties:
    """Tests for Halfspace mathematical properties."""

    def test_is_smooth(self):
        """Test that Halfspace is not smooth (indicator function)."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = 1.0
        halfspace = Halfspace(x, c=c, upper=upper)

        assert halfspace.is_smooth() is False

    def test_is_proxable(self):
        """Test that Halfspace is proxable (has projection)."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = 1.0
        halfspace = Halfspace(x, c=c, upper=upper)

        assert halfspace.is_proxable() is True

    def test_is_subsamplable(self):
        """Test that Halfspace is not subsamplable (constraint)."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = 1.0
        halfspace = Halfspace(x, c=c, upper=upper)

        assert halfspace.is_subsamplable() is False

    def test_subsample_raises_error(self):
        """Test that subsample raises NotImplementedError."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = 1.0
        halfspace = Halfspace(x, c=c, upper=upper)
        indices = torch.tensor([0, 1, 2])

        with pytest.raises(NotImplementedError, match="not subsamplable"):
            halfspace.subsample(indices)


# ===============================
# Test Forward (Inherited from Polyhedron)
# ===============================


class TestHalfspaceForward:
    """Tests for Halfspace forward evaluation."""

    def test_forward_satisfied(self):
        """Test forward when halfspace constraint is satisfied."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # c^T x = [1, 1, 1] · [1, 2, 3] = 6 <= 10
        c = torch.ones(3)
        upper = 10.0
        halfspace = Halfspace(x, c=c, upper=upper)

        result = halfspace.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_violated(self):
        """Test forward when halfspace constraint is violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # c^T x = [1, 1, 1] · [1, 2, 3] = 6 > 5
        c = torch.ones(3)
        upper = 5.0
        halfspace = Halfspace(x, c=c, upper=upper)

        result = halfspace.forward()

        assert torch.isinf(result)

    def test_forward_at_boundary(self):
        """Test forward when value is exactly at boundary."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # c^T x = 6 <= 6 (exactly at boundary)
        c = torch.ones(3)
        upper = 6.0
        halfspace = Halfspace(x, c=c, upper=upper)

        result = halfspace.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_negative_values(self):
        """Test forward with negative values."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([-1.0, -2.0, -3.0])

        # c^T x = [1, 1, 1] · [-1, -2, -3] = -6 <= 0
        c = torch.ones(3)
        upper = 0.0
        halfspace = Halfspace(x, c=c, upper=upper)

        result = halfspace.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_axis_aligned(self):
        """Test forward with axis-aligned halfspace."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([2.0, 5.0, 3.0])

        # c^T x = [1, 0, 0] · [2, 5, 3] = 2 <= 5 (x[0] <= 5)
        c = torch.tensor([1.0, 0.0, 0.0])
        upper = 5.0
        halfspace = Halfspace(x, c=c, upper=upper)

        result = halfspace.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Proximal Operator (Projection)
# ===============================


class TestHalfspaceProx:
    """Tests for Halfspace proximal operator (projection)."""

    def test_prox_inside_halfspace(self):
        """Test prox when point is already inside halfspace."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(10.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # c^T location = 6 <= 10 (already satisfied)
        location = torch.tensor([1.0, 2.0, 3.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should return unchanged
        assert torch.allclose(result, location)

    def test_prox_violating_halfspace(self):
        """Test prox when point violates halfspace."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # c^T location = 9 > 5 (violated)
        location = torch.tensor([3.0, 3.0, 3.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should project onto boundary: c^T result = 5
        assert torch.allclose(torch.dot(c, result), torch.tensor(5.0), atol=1e-5)

        # Projection should be along normal direction
        diff = location - result
        normalized_diff = diff / torch.linalg.norm(diff)
        normalized_c = c / torch.linalg.norm(c)
        assert torch.allclose(normalized_diff, normalized_c, atol=1e-5)

    def test_prox_exactly_at_boundary(self):
        """Test prox when point is exactly at boundary."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(6.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # c^T location = 6 <= 6 (exactly at boundary)
        location = torch.tensor([2.0, 2.0, 2.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should return unchanged (or very close)
        assert torch.allclose(result, location, atol=1e-6)

    def test_prox_perpendicular_projection(self):
        """Test that projection is perpendicular to boundary."""
        x = Variable((2,), name="x")
        c = torch.tensor([1.0, 0.0])  # x-axis normal
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # Point at (2, 5) should project to (1, 5)
        location = torch.tensor([2.0, 5.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        expected = torch.tensor([1.0, 5.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_prox_with_negative_c(self):
        """Test prox with negative normal vector."""
        x = Variable((3,), name="x")
        c = -torch.ones(3)  # Flipped normal
        upper = torch.tensor(-5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # c^T location = -9 <= -5 (satisfied)
        location = torch.tensor([3.0, 3.0, 3.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should return unchanged
        assert torch.allclose(result, location, atol=1e-5)

    def test_prox_ignores_scaling(self):
        """Test that prox_scaling doesn't affect projection."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.tensor([3.0, 3.0, 3.0])

        # Different scaling should give same result
        result1 = halfspace.prox(location, prox_scaling=0.5)
        result2 = halfspace.prox(location, prox_scaling=1.0)
        result3 = halfspace.prox(location, prox_scaling=10.0)

        assert torch.allclose(result1, result2, atol=1e-5)
        assert torch.allclose(result2, result3, atol=1e-5)

    def test_prox_unit_normal_vector(self):
        """Test prox with unit normal vector."""
        x = Variable((3,), name="x")
        c = torch.tensor([1.0, 0.0, 0.0]) / math.sqrt(1.0)  # Already unit
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.tensor([2.0, 3.0, 4.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should project x-coordinate to 1, keep others
        expected = torch.tensor([1.0, 3.0, 4.0])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_prox_scaled_normal_vector(self):
        """Test prox with scaled normal vector."""
        x = Variable((3,), name="x")
        c = torch.ones(3) * 2.0  # Scaled by 2
        upper = torch.tensor(10.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # c^T location = 2*9 = 18 > 10
        location = torch.tensor([3.0, 3.0, 3.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should project to boundary: c^T result = 10
        assert torch.allclose(torch.dot(c, result), torch.tensor(10.0), atol=1e-5)

    def test_prox_preserves_shape(self):
        """Test that prox preserves input shape."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.randn(5)
        result = halfspace.prox(location, prox_scaling=1.0)

        assert result.shape == location.shape

    def test_prox_is_idempotent(self):
        """Test that prox is idempotent (projecting twice = projecting once)."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.tensor([3.0, 3.0, 3.0])

        result1 = halfspace.prox(location, prox_scaling=1.0)
        result2 = halfspace.prox(result1, prox_scaling=1.0)

        # Projecting twice should give same result as once
        assert torch.allclose(result1, result2, atol=1e-5)

    def test_prox_reduces_violation(self):
        """Test that prox reduces constraint violation."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.tensor([5.0, 5.0, 5.0])  # c^T x = 15 >> 5
        result = halfspace.prox(location, prox_scaling=1.0)

        # Result should satisfy constraint
        assert torch.dot(c, result) <= upper + 1e-5

    def test_prox_2d_geometry(self):
        """Test prox with clear 2D geometry."""
        x = Variable((2,), name="x")
        # Halfspace: x + y <= 1 (normal = [1, 1], upper = 1)
        c = torch.ones(2)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # Point (2, 2) violates: 2 + 2 = 4 > 1
        location = torch.tensor([2.0, 2.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should project to boundary
        assert torch.allclose(torch.dot(c, result), upper, atol=1e-5)

        # Projection should move along diagonal (normal direction)
        # From (2,2) to boundary at distance d along [-1,-1]/sqrt(2)
        # Result should be (2,2) - d*(1,1)/sqrt(2) where d*sqrt(2) = 3
        # So d = 3/sqrt(2), result ≈ (0.5, 0.5)
        assert torch.allclose(result, torch.tensor([0.5, 0.5]), atol=1e-4)


# ===============================
# Test Integration & Use Cases
# ===============================


class TestHalfspaceIntegration:
    """Integration tests for Halfspace in realistic scenarios."""

    def test_halfspace_as_upper_bound(self):
        """Test Halfspace as simple upper bound constraint."""
        x = Variable((1,), name="x")
        # x <= 5  →  1*x <= 5
        c = torch.ones(1)
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # Project 10 to 5
        location = torch.tensor([10.0])
        projected = halfspace.prox(location, prox_scaling=1.0)

        assert torch.allclose(projected, torch.tensor([5.0]), atol=1e-5)

    def test_halfspace_budget_constraint(self):
        """Test Halfspace for budget constraint (prices · quantities <= budget)."""
        x = Variable((3,), name="x")  # quantities
        prices = torch.tensor([2.0, 3.0, 5.0])
        budget = torch.tensor(20.0)
        halfspace = Halfspace(x, c=prices, upper=budget)

        # Spending = [1, 2, 3] · [2, 3, 5] = 2 + 6 + 15 = 23 > 20
        quantities = torch.tensor([1.0, 2.0, 3.0])
        projected = halfspace.prox(quantities, prox_scaling=1.0)

        # Should reduce spending to budget
        spending = torch.dot(prices, projected)
        assert spending <= budget + 1e-5

    def test_halfspace_linear_inequality(self):
        """Test Halfspace for general linear inequality."""
        x = Variable((4,), name="x")
        c = torch.tensor([1.0, 2.0, 3.0, 4.0])
        upper = torch.tensor(10.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # Test with various points
        points = [
            torch.tensor([1.0, 1.0, 1.0, 1.0]),  # Sum = 10, satisfied
            torch.tensor([2.0, 2.0, 2.0, 2.0]),  # Sum = 20, violated
            torch.tensor([0.0, 0.0, 0.0, 0.0]),  # Sum = 0, satisfied
        ]

        for point in points:
            projected = halfspace.prox(point, prox_scaling=1.0)
            assert torch.dot(c, projected) <= upper + 1e-5

    def test_halfspace_state_dict(self):
        """Test Halfspace state dict save and load."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        state = halfspace.state_dict()

        assert "C" in state
        assert "upper" in state
        assert "x" in state

    def test_halfspace_parameter_tracking(self):
        """Test that Halfspace tracks variable parameters."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        params = list(halfspace.parameters())
        assert len(params) == 1
        assert params[0].shape == torch.Size([5])

    def test_halfspace_buffer_tracking(self):
        """Test that Halfspace stores c and upper as buffers."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        buffers = dict(halfspace.named_buffers())
        assert "C" in buffers
        assert "upper" in buffers

    def test_halfspace_device_transfer(self):
        """Test moving Halfspace to device."""
        x = Variable((5,), name="x", device=torch.device("cpu"))
        c = torch.randn(5)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        halfspace = halfspace.to("cpu")

        assert halfspace.C.device.type == "cpu"
        assert halfspace.upper.device.type == "cpu"

    def test_halfspace_dtype_conversion(self):
        """Test Halfspace dtype conversion."""
        x = Variable((5,), name="x", dtype=torch.float32)
        c = torch.randn(5, dtype=torch.float32)
        upper = torch.tensor(1.0, dtype=torch.float32)
        halfspace = Halfspace(x, c=c, upper=upper)

        halfspace = halfspace.to(torch.float64)

        assert halfspace.get_variable("x").dtype == torch.float64


# ===============================
# Test Edge Cases
# ===============================


class TestHalfspaceEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_normal_vector(self):
        """Test with very small (but non-zero) normal vector."""
        x = Variable((3,), name="x")
        c = torch.ones(3) * 1e-10
        upper = torch.tensor(1e-9)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.ones(3)
        # Should still compute projection (though numerically sensitive)
        result = halfspace.prox(location, prox_scaling=1.0)

        assert result.shape == location.shape

    def test_very_large_normal_vector(self):
        """Test with very large normal vector."""
        x = Variable((3,), name="x")
        c = torch.ones(3) * 1e6
        upper = torch.tensor(1e7)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.ones(3) * 2.0
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should handle large scale
        assert torch.dot(c, result) <= upper + 1e-1  # Looser tolerance for large scale

    def test_single_dimension(self):
        """Test with single-dimensional variable."""
        x = Variable((1,), name="x")
        c = torch.tensor([1.0])
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.tensor([10.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        assert torch.allclose(result, torch.tensor([5.0]), atol=1e-5)

    def test_very_large_violation(self):
        """Test with point far outside halfspace."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(1.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # Point very far outside
        location = torch.ones(3) * 1000.0
        result = halfspace.prox(location, prox_scaling=1.0)

        # Should still project correctly
        assert torch.allclose(torch.dot(c, result), upper, atol=1e-3)

    def test_prox_with_negative_location(self):
        """Test prox with entirely negative location."""
        x = Variable((3,), name="x")
        c = torch.ones(3)
        upper = torch.tensor(0.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        # c^T location = -6 <= 0 (satisfied)
        location = -torch.ones(3) * 2.0
        result = halfspace.prox(location, prox_scaling=1.0)

        assert torch.allclose(result, location, atol=1e-5)

    def test_orthogonal_projection_property(self):
        """Test that projection satisfies orthogonality property."""
        x = Variable((3,), name="x")
        c = torch.tensor([1.0, 2.0, 3.0])
        upper = torch.tensor(5.0)
        halfspace = Halfspace(x, c=c, upper=upper)

        location = torch.tensor([5.0, 5.0, 5.0])
        result = halfspace.prox(location, prox_scaling=1.0)

        # The difference should be parallel to c
        diff = location - result
        if torch.linalg.norm(diff) > 1e-6:  # Only if projection moved
            # diff should be proportional to c
            ratio = diff / c
            assert torch.allclose(ratio[0], ratio[1], atol=1e-4)
            assert torch.allclose(ratio[1], ratio[2], atol=1e-4)


# ===============================
# Test Documentation Examples
# ===============================


class TestDocumentationExamples:
    """Tests for examples from docstrings."""

    def test_basic_halfspace_example(self):
        """Test basic halfspace example from docstring."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        halfspace = Halfspace(x, c=c, upper=1.0)

        assert isinstance(halfspace, Halfspace)
        assert torch.equal(halfspace.C, c)
        assert torch.allclose(halfspace.upper, torch.tensor(1.0))

    def test_nonnegative_coordinate_example(self):
        """Test non-negativity example from docstring."""
        x = Variable((5,), name="x")
        # x[0] >= 0  →  -x[0] <= 0
        c = torch.zeros(5)
        c[0] = -1.0
        nonneg = Halfspace(x, c=c, upper=0.0)

        assert isinstance(nonneg, Halfspace)
        assert c[0] == -1.0
        assert torch.allclose(nonneg.upper, torch.tensor(0.0))

    def test_prox_example(self):
        """Test proximal operator example from docstring."""
        x = Variable((5,), name="x")
        c = torch.randn(5)
        halfspace = Halfspace(x, c=c, upper=1.0)

        violating_point = torch.randn(5)
        projected = halfspace.prox(violating_point, prox_scaling=1.0)

        # Projected point should satisfy constraint (or be close)
        assert torch.dot(c, projected) <= halfspace.upper + 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
