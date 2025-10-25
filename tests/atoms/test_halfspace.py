"""Tests for Halfspace atom."""

import pytest
import torch

from rlaopt.atoms.halfspace import Halfspace
from rlaopt.expression.variable import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="x")


@pytest.fixture
def small_vector_var():
    """Create a small vector variable."""
    return Variable((3,), name="x")


@pytest.fixture
def simple_halfspace_data():
    """Create simple halfspace data: sum(x) <= 5."""
    return {"c": torch.ones(5), "upper": 5.0}


@pytest.fixture
def coordinate_halfspace_data():
    """Create coordinate halfspace: x[0] <= 1."""
    c = torch.zeros(5)
    c[0] = 1.0
    return {"c": c, "upper": 1.0}


class TestHalfspace:
    """Test Halfspace constraint atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_halfspace_constraint(self, vector_var, simple_halfspace_data):
        """Test initialization with halfspace constraint."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        assert halfspace.C is not None
        assert halfspace.upper is not None
        assert halfspace.A is None
        assert halfspace.b is None

    def test_init_stores_constraint_correctly(
        self, vector_var, coordinate_halfspace_data
    ):
        """Test initialization stores c and upper correctly."""
        halfspace = Halfspace(vector_var, **coordinate_halfspace_data)
        assert torch.equal(halfspace.C, coordinate_halfspace_data["c"])
        assert halfspace.upper == coordinate_halfspace_data["upper"]

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_zero_when_satisfied(self, small_vector_var):
        """Test forward() returns 0 when constraint is satisfied."""
        halfspace = Halfspace(small_vector_var, c=torch.ones(3), upper=5.0)
        small_vector_var.value.data = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3 <= 5
        result = halfspace.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_zero_at_boundary(self, small_vector_var):
        """Test forward() returns 0 when exactly at boundary."""
        halfspace = Halfspace(small_vector_var, c=torch.ones(3), upper=3.0)
        small_vector_var.value.data = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3 = upper
        result = halfspace.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_inf_when_violated(self, small_vector_var):
        """Test forward() returns infinity when constraint is violated."""
        halfspace = Halfspace(small_vector_var, c=torch.ones(3), upper=2.0)
        small_vector_var.value.data = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3 > 2
        result = halfspace.forward()
        assert torch.isinf(result)

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_false(self, vector_var, simple_halfspace_data):
        """Test halfspace is not smooth (inherits from Polyhedron)."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        assert halfspace.is_smooth() is False

    def test_is_proxable_returns_true(self, vector_var, simple_halfspace_data):
        """Test halfspace is proxable."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        assert halfspace.is_proxable() is True

    # ----------------------
    # Proximal operator tests
    # ----------------------

    def test_prox_returns_identity_for_feasible_point(
        self, vector_var, simple_halfspace_data
    ):
        """Test prox() returns input unchanged if already feasible."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        location = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])  # Sum = 2.5 <= 5

        result = halfspace.prox(location, prox_scaling=1.0)
        assert torch.allclose(result, location)

    def test_prox_projects_infeasible_point(self, vector_var, simple_halfspace_data):
        """Test prox() projects infeasible point onto halfspace boundary."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        location = torch.ones(5) * 2  # Sum = 10 > 5, violates constraint

        result = halfspace.prox(location, prox_scaling=1.0)

        # Check result satisfies constraint
        assert torch.dot(halfspace.C, result) <= halfspace.upper + 1e-5

    def test_prox_moves_perpendicular_to_boundary(self, small_vector_var):
        """Test prox() moves point perpendicular to halfspace boundary."""
        c = torch.tensor([1.0, 0.0, 0.0])
        halfspace = Halfspace(small_vector_var, c=c, upper=1.0)
        location = torch.tensor([5.0, 2.0, 3.0])  # x[0] = 5 > 1, violates

        result = halfspace.prox(location, prox_scaling=1.0)

        # Only first coordinate should change (perpendicular to boundary)
        assert torch.allclose(result[1:], location[1:])
        assert result[0] <= 1.0 + 1e-5

    def test_prox_at_boundary_unchanged(self, vector_var, simple_halfspace_data):
        """Test prox() of point on boundary returns same point."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        location = torch.ones(5)  # Sum = 5 = upper (on boundary)

        result = halfspace.prox(location, prox_scaling=1.0)
        assert torch.allclose(result, location, atol=1e-5)

    def test_prox_independent_of_scaling(self, vector_var, simple_halfspace_data):
        """Test prox() is independent of prox_scaling parameter."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        location = torch.ones(5) * 2  # Sum = 10 > 5

        result1 = halfspace.prox(location, prox_scaling=1.0)
        result2 = halfspace.prox(location, prox_scaling=100.0)

        # Projection should be independent of scaling
        assert torch.allclose(result1, result2)

    def test_prox_with_coordinate_constraint(
        self, vector_var, coordinate_halfspace_data
    ):
        """Test prox() with single coordinate constraint."""
        halfspace = Halfspace(vector_var, **coordinate_halfspace_data)
        location = torch.tensor([5.0, 2.0, 3.0, 4.0, 1.0])  # x[0] = 5 > 1

        result = halfspace.prox(location, prox_scaling=1.0)

        # First coordinate should be clamped to 1.0
        assert torch.allclose(result[0], torch.tensor(1.0), atol=1e-5)
        # Others unchanged
        assert torch.allclose(result[1:], location[1:])

    def test_prox_minimizes_distance(self, vector_var, simple_halfspace_data):
        """Test prox() finds closest feasible point."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        location = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])  # Sum = 10 > 5

        result = halfspace.prox(location, prox_scaling=1.0)

        # Result should satisfy constraint
        assert torch.dot(halfspace.C, result) <= halfspace.upper + 1e-5

        # Should be close to original (minimizes distance)
        distance = torch.norm(result - location)
        assert distance > 0  # Must move
        assert distance < torch.norm(location)  # But not too far

    # ----------------------
    # Edge cases
    # ----------------------

    def test_with_negative_upper_bound(self, small_vector_var):
        """Test halfspace with negative upper bound."""
        halfspace = Halfspace(small_vector_var, c=torch.ones(3), upper=-1.0)
        location = torch.zeros(3)  # Sum = 0 > -1, violates

        result = halfspace.prox(location, prox_scaling=1.0)
        assert torch.dot(halfspace.C, result) <= halfspace.upper + 1e-5

    def test_with_scaled_normal_vector(self, small_vector_var):
        """Test halfspace with non-unit normal vector."""
        c = torch.tensor([2.0, 2.0, 2.0])  # Scaled version of [1,1,1]
        halfspace = Halfspace(small_vector_var, c=c, upper=6.0)
        location = torch.ones(3) * 2  # c^T x = 12 > 6

        result = halfspace.prox(location, prox_scaling=1.0)
        assert torch.dot(c, result) <= 6.0 + 1e-5

    def test_gradient_flows_through_prox(self, vector_var, simple_halfspace_data):
        """Test gradients flow through prox operator."""
        halfspace = Halfspace(vector_var, **simple_halfspace_data)
        location = torch.ones(5) * 2  # Infeasible
        location.requires_grad_(True)

        result = halfspace.prox(location, prox_scaling=1.0)
        loss = result.sum()
        loss.backward()

        assert location.grad is not None
        assert not torch.isnan(location.grad).any()
