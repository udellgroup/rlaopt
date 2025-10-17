"""Comprehensive tests for the Box constraint atom."""

import pytest
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.atoms.box import Box
from rlaopt.atoms.polyhedron import Polyhedron
from rlaopt.expression import Variable

# ===============================
# Test Initialization
# ===============================


class TestBoxInit:
    """Tests for Box initialization."""

    def test_init_with_lower_and_upper(self):
        """Test initialization with both lower and upper bounds."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        assert isinstance(box, Box)
        assert isinstance(box, Polyhedron)
        assert isinstance(box, AtomExpression)

    def test_init_with_tensor_bounds(self):
        """Test initialization with tensor bounds."""
        x = Variable((5,), name="x")
        lower = torch.zeros(5)
        upper = torch.ones(5)

        box = Box(x, lower=lower, upper=upper)

        assert torch.equal(box.lower, lower)
        assert torch.equal(box.upper, upper)

    def test_init_with_float_bounds(self):
        """Test initialization with float bounds."""
        x = Variable((5,), name="x")
        box = Box(x, lower=-1.0, upper=1.0)

        assert isinstance(box.lower, torch.Tensor)
        assert isinstance(box.upper, torch.Tensor)

    def test_init_lower_only(self):
        """Test initialization with only lower bound."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0)

        assert torch.equal(box.lower, torch.tensor(0.0))
        assert torch.isinf(box.upper)

    def test_init_upper_only(self):
        """Test initialization with only upper bound."""
        x = Variable((5,), name="x")
        box = Box(x, upper=1.0)

        assert torch.isinf(box.lower) and box.lower < 0
        assert torch.equal(box.upper, torch.tensor(1.0))

    def test_init_no_bounds_raises_error(self):
        """Test that no bounds raises ValueError."""
        x = Variable((5,), name="x")

        with pytest.raises(ValueError, match="trivial polyhedron"):
            Box(x)

    def test_init_negative_bounds(self):
        """Test initialization with negative bounds."""
        x = Variable((5,), name="x")
        box = Box(x, lower=-10.0, upper=-1.0)

        assert box.lower.item() == -10.0
        assert box.upper.item() == -1.0

    def test_init_different_variable_shapes(self):
        """Test initialization with different variable shapes."""
        shapes = [(5,), (10,), (3, 4), (2, 3, 4)]

        for shape in shapes:
            x = Variable(shape, name="x")
            box = Box(x, lower=0.0, upper=1.0)
            assert box.var_name == "x"


# ===============================
# Test Properties
# ===============================


class TestBoxProperties:
    """Tests for Box mathematical properties."""

    def test_is_smooth(self):
        """Test that Box is not smooth (indicator function)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        assert box.is_smooth() is False

    def test_is_proxable(self):
        """Test that Box is proxable (has projection)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        assert box.is_proxable() is True

    def test_is_subsamplable(self):
        """Test that Box is not subsamplable (constraint)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        assert box.is_subsamplable() is False

    def test_subsample_raises_error(self):
        """Test that subsample raises NotImplementedError."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)
        indices = torch.tensor([0, 1, 2])

        with pytest.raises(NotImplementedError, match="not subsamplable"):
            box.subsample(indices)


# ===============================
# Test Forward (Inherited from Polyhedron)
# ===============================


class TestBoxForward:
    """Tests for Box forward evaluation."""

    def test_forward_satisfied(self):
        """Test forward when box constraints are satisfied."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.9])
        box = Box(x, lower=0.0, upper=1.0)

        result = box.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_violated_upper(self):
        """Test forward when upper bound is violated."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.3, 1.2, 0.2, 0.9])  # 1.2 > 1.0
        box = Box(x, lower=0.0, upper=1.0)

        result = box.forward()

        assert torch.isinf(result)

    def test_forward_violated_lower(self):
        """Test forward when lower bound is violated."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, -0.1, 0.7, 0.2, 0.9])  # -0.1 < 0.0
        box = Box(x, lower=0.0, upper=1.0)

        result = box.forward()

        assert torch.isinf(result)

    def test_forward_at_boundaries(self):
        """Test forward when values are at boundaries."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0])
        box = Box(x, lower=0.0, upper=1.0)

        result = box.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_lower_bound_only(self):
        """Test forward with only lower bound."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([10.0, 100.0, 0.0, 5.0, 50.0])
        box = Box(x, lower=0.0)

        result = box.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_upper_bound_only(self):
        """Test forward with only upper bound."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([-100.0, -10.0, 0.5, 0.0, -50.0])
        box = Box(x, upper=1.0)

        result = box.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Proximal Operator (Projection)
# ===============================


class TestBoxProx:
    """Tests for Box proximal operator (projection/clamping)."""

    def test_prox_inside_box(self):
        """Test prox when point is already inside box."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.9])
        result = box.prox(location, prox_scaling=1.0)

        # Should return unchanged (already feasible)
        assert torch.equal(result, location)

    def test_prox_above_upper_bound(self):
        """Test prox when point is above upper bound."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([1.5, 2.0, 0.5, 1.2, 0.8])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([1.0, 1.0, 0.5, 1.0, 0.8])
        assert torch.equal(result, expected)

    def test_prox_below_lower_bound(self):
        """Test prox when point is below lower bound."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([-0.5, 0.5, -1.0, 0.2, -0.1])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([0.0, 0.5, 0.0, 0.2, 0.0])
        assert torch.equal(result, expected)

    def test_prox_mixed_violations(self):
        """Test prox with mixed violations (above and below)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([-0.5, 1.5, 0.5, -1.0, 2.0])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0])
        assert torch.equal(result, expected)

    def test_prox_at_boundaries(self):
        """Test prox when point is exactly at boundaries."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0])
        result = box.prox(location, prox_scaling=1.0)

        assert torch.equal(result, location)

    def test_prox_ignores_scaling(self):
        """Test that prox_scaling doesn't affect projection."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([-0.5, 1.5, 0.5, 2.0, -1.0])

        # Different scaling should give same result
        result1 = box.prox(location, prox_scaling=0.5)
        result2 = box.prox(location, prox_scaling=1.0)
        result3 = box.prox(location, prox_scaling=10.0)

        assert torch.equal(result1, result2)
        assert torch.equal(result2, result3)

    def test_prox_negative_bounds(self):
        """Test prox with negative bounds."""
        x = Variable((5,), name="x")
        box = Box(x, lower=-2.0, upper=-1.0)

        location = torch.tensor([-0.5, -1.5, -3.0, -1.0, -2.5])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([-1.0, -1.5, -2.0, -1.0, -2.0])
        assert torch.equal(result, expected)

    def test_prox_asymmetric_bounds(self):
        """Test prox with asymmetric bounds."""
        x = Variable((5,), name="x")
        box = Box(x, lower=-1.0, upper=2.0)

        location = torch.tensor([-2.0, 0.0, 3.0, 1.0, -0.5])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([-1.0, 0.0, 2.0, 1.0, -0.5])
        assert torch.equal(result, expected)

    def test_prox_lower_bound_only(self):
        """Test prox with only lower bound (upper = inf)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0)

        location = torch.tensor([-1.0, 5.0, -10.0, 100.0, 0.0])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([0.0, 5.0, 0.0, 100.0, 0.0])
        assert torch.equal(result, expected)

    def test_prox_upper_bound_only(self):
        """Test prox with only upper bound (lower = -inf)."""
        x = Variable((5,), name="x")
        box = Box(x, upper=1.0)

        location = torch.tensor([2.0, -5.0, 0.5, -100.0, 1.5])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([1.0, -5.0, 0.5, -100.0, 1.0])
        assert torch.equal(result, expected)

    def test_prox_multidimensional(self):
        """Test prox with multidimensional variable."""
        x = Variable((3, 4), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor(
            [[-0.5, 0.5, 1.5, 0.3], [0.7, -0.2, 1.2, 0.9], [0.1, 0.4, -1.0, 2.0]]
        )

        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor(
            [[0.0, 0.5, 1.0, 0.3], [0.7, 0.0, 1.0, 0.9], [0.1, 0.4, 0.0, 1.0]]
        )

        assert torch.equal(result, expected)

    def test_prox_preserves_shape(self):
        """Test that prox preserves input shape."""
        x = Variable((3, 4), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.randn(3, 4)
        result = box.prox(location, prox_scaling=1.0)

        assert result.shape == location.shape

    def test_prox_is_idempotent(self):
        """Test that prox is idempotent (projecting twice = projecting once)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([-1.0, 0.5, 2.0, 0.3, -0.5])

        result1 = box.prox(location, prox_scaling=1.0)
        result2 = box.prox(result1, prox_scaling=1.0)

        # Projecting twice should give same result as once
        assert torch.equal(result1, result2)

    def test_prox_with_vector_bounds(self):
        """Test prox with different bounds for each element."""
        x = Variable((5,), name="x")
        lower = torch.tensor([0.0, -1.0, -2.0, 0.5, -0.5])
        upper = torch.tensor([1.0, 1.0, 2.0, 1.5, 0.5])
        box = Box(x, lower=lower, upper=upper)

        location = torch.tensor([-1.0, 0.0, 3.0, 1.0, 0.0])
        result = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([0.0, 0.0, 2.0, 1.0, 0.0])
        assert torch.equal(result, expected)


# ===============================
# Test Integration & Use Cases
# ===============================


class TestBoxIntegration:
    """Integration tests for Box in realistic scenarios."""

    def test_box_as_nonnegative_constraint(self):
        """Test Box for non-negativity constraint (x >= 0)."""
        x = Variable((10,), name="x")
        box = Box(x, lower=0.0)

        # Project negative values to zero
        negative = torch.tensor([-1.0, -0.5, 1.0, 2.0, -2.0, 3.0, -0.1, 0.5, -5.0, 0.0])
        projected = box.prox(negative, prox_scaling=1.0)

        # All values should be >= 0
        assert torch.all(projected >= 0)

    def test_box_as_probability_constraint(self):
        """Test Box for probability bounds (0 <= p <= 1)."""
        p = Variable((5,), name="p")
        box = Box(p, lower=0.0, upper=1.0)

        # Project to valid probability range
        invalid = torch.tensor([-0.2, 0.5, 1.3, 0.8, 2.0])
        valid = box.prox(invalid, prox_scaling=1.0)

        assert torch.all((valid >= 0) & (valid <= 1))

    def test_box_in_optimization_loop(self):
        """Test Box projection in optimization loop."""
        x = Variable((5,), name="x")
        x.value.data = torch.randn(5)
        box = Box(x, lower=0.0, upper=1.0)

        # Simulate optimization step with projection
        for _ in range(10):
            # Gradient step (might go out of bounds)
            x.value.data = x.value.data - 0.1 * torch.randn(5)

            # Project back to feasible set
            x.value.data = box.prox(x.value.data, prox_scaling=1.0)

        # Final value should be in box
        assert torch.all((x.value >= 0) & (x.value <= 1))

    def test_box_gradient_clipping(self):
        """Test Box for gradient clipping."""
        x = Variable((5,), name="x")
        box = Box(x, lower=-1.0, upper=1.0)

        # Large gradients
        gradients = torch.tensor([-5.0, 3.0, -2.0, 10.0, -8.0])

        # Clip to [-1, 1]
        clipped = box.prox(gradients, prox_scaling=1.0)

        assert torch.all((clipped >= -1) & (clipped <= 1))
        assert torch.equal(clipped, torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0]))

    def test_box_state_dict(self):
        """Test Box state dict save and load."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        state = box.state_dict()

        assert "lower" in state
        assert "upper" in state
        assert "x" in state

    def test_box_parameter_tracking(self):
        """Test that Box tracks variable parameters."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        params = list(box.parameters())
        assert len(params) == 1
        assert params[0].shape == torch.Size([5])

    def test_box_buffer_tracking(self):
        """Test that Box stores bounds as buffers."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        buffers = dict(box.named_buffers())
        assert "lower" in buffers
        assert "upper" in buffers

    def test_box_device_transfer(self):
        """Test moving Box to device."""
        x = Variable((5,), name="x", device=torch.device("cpu"))
        box = Box(x, lower=0.0, upper=1.0)

        box = box.to("cpu")

        assert box.lower.device.type == "cpu"
        assert box.upper.device.type == "cpu"

    def test_box_dtype_conversion(self):
        """Test Box dtype conversion."""
        x = Variable((5,), name="x", dtype=torch.float32)
        box = Box(x, lower=0.0, upper=1.0)

        box = box.to(torch.float64)

        assert box.get_variable("x").dtype == torch.float64


# ===============================
# Test Edge Cases
# ===============================


class TestBoxEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_variable(self):
        """Test Box with empty variable."""
        x = Variable((0,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        result = box.forward()
        assert torch.allclose(result, torch.tensor(0.0))

        location = torch.tensor([])
        projected = box.prox(location, prox_scaling=1.0)
        assert projected.shape == torch.Size([0])

    def test_single_element_variable(self):
        """Test Box with single element."""
        x = Variable((1,), name="x")
        x.value.data = torch.tensor([0.5])
        box = Box(x, lower=0.0, upper=1.0)

        result = box.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_very_large_bounds(self):
        """Test Box with very large bounds."""
        x = Variable((5,), name="x")
        box = Box(x, lower=-1e10, upper=1e10)

        location = torch.tensor([1e9, -1e9, 0.0, 5e9, -5e9])
        projected = box.prox(location, prox_scaling=1.0)

        # Should be unchanged (within bounds)
        assert torch.equal(projected, location)

    def test_very_small_bounds(self):
        """Test Box with very small bounds (tight constraint)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1e-10)

        location = torch.ones(5)
        projected = box.prox(location, prox_scaling=1.0)

        # Should all be clamped to 1e-10
        assert torch.all(projected <= 1e-10)

    def test_equal_bounds(self):
        """Test Box where lower equals upper (equality constraint)."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.5, upper=0.5)

        location = torch.randn(5)
        projected = box.prox(location, prox_scaling=1.0)

        # All elements should be exactly 0.5
        assert torch.allclose(projected, torch.full((5,), 0.5))

    def test_prox_with_nan(self):
        """Test prox behavior with NaN values."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([0.5, float("nan"), 0.7, 0.3, 0.9])
        projected = box.prox(location, prox_scaling=1.0)

        # NaN should remain NaN after clamping
        assert torch.isnan(projected[1])

    def test_prox_with_inf(self):
        """Test prox behavior with infinity values."""
        x = Variable((5,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        location = torch.tensor([float("inf"), 0.5, float("-inf"), 0.7, 2.0])
        projected = box.prox(location, prox_scaling=1.0)

        expected = torch.tensor([1.0, 0.5, 0.0, 0.7, 1.0])
        assert torch.equal(projected, expected)


# ===============================
# Test Documentation Examples
# ===============================


class TestDocumentationExamples:
    """Tests for examples from docstrings."""

    def test_standard_box_example(self):
        """Test standard box constraint example from docstring."""
        x = Variable((10,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        assert isinstance(box, Box)
        assert box.lower.item() == 0.0
        assert box.upper.item() == 1.0

    def test_nonnegative_example(self):
        """Test non-negativity example from docstring."""
        x = Variable((10,), name="x")
        box_nonneg = Box(x, lower=0.0)

        assert isinstance(box_nonneg, Box)
        assert box_nonneg.lower.item() == 0.0
        assert torch.isinf(box_nonneg.upper)

    def test_upper_bound_example(self):
        """Test upper bound example from docstring."""
        x = Variable((10,), name="x")
        box_upper = Box(x, upper=1.0)

        assert isinstance(box_upper, Box)
        assert torch.isinf(box_upper.lower) and box_upper.lower < 0
        assert box_upper.upper.item() == 1.0

    def test_prox_example(self):
        """Test proximal operator example from docstring."""
        x = Variable((10,), name="x")
        box = Box(x, lower=0.0, upper=1.0)

        out_of_bounds = torch.randn(10)
        projected = box.prox(out_of_bounds, prox_scaling=1.0)

        # Projected values should be in [0, 1]
        assert torch.all((projected >= 0) & (projected <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
