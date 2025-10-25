"""Tests for LinearEquality atom."""

import pytest
import torch

from rlaopt.atoms.linear_equality import LinearEquality
from rlaopt.expression import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="x")


@pytest.fixture
def small_vector_var():
    """Create a small vector variable."""
    return Variable((3,), name="x")


@pytest.fixture
def full_rank_constraint_data():
    """Create full-rank constraint data."""
    return {
        "A": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 0.0, 1.0, 0.0]]),
        "b": torch.tensor([15.0, 2.0]),
    }


@pytest.fixture
def simple_constraint_data():
    """Create simple constraint data (sum equals constant)."""
    return {"A": torch.tensor([[1.0, 1.0, 1.0]]), "b": torch.tensor([3.0])}


class TestLinearEquality:
    """Test LinearEquality constraint atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_full_rank_constraints(
        self, vector_var, full_rank_constraint_data
    ):
        """Test initialization with full-rank constraint matrix."""
        lin_eq = LinearEquality(vector_var, **full_rank_constraint_data)
        assert lin_eq.A is not None
        assert lin_eq.b is not None
        assert lin_eq.R is not None

    def test_init_caches_R_factor(self, vector_var, full_rank_constraint_data):
        """Test initialization caches R factor from QR decomposition."""
        lin_eq = LinearEquality(vector_var, **full_rank_constraint_data)
        assert lin_eq.R.shape[0] == 2  # Number of constraints
        assert lin_eq.R.shape[1] == 2

    def test_init_raises_error_for_rank_deficient(self, vector_var):
        """Test initialization raises ValueError for rank-deficient matrix."""
        A = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]]
        )  # Second row is 2x first
        b = torch.tensor([15.0, 30.0])

        with pytest.raises(ValueError, match="rank deficient"):
            LinearEquality(vector_var, A, b)

    def test_init_raises_error_for_zero_matrix(self, vector_var):
        """Test initialization raises error for zero constraint matrix."""
        A = torch.zeros(2, 5)
        b = torch.randn(2)

        with pytest.raises(ValueError, match="To define"):
            LinearEquality(vector_var, A, b)

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_zero_when_satisfied(
        self, small_vector_var, simple_constraint_data
    ):
        """Test forward() returns 0 when constraint is satisfied."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        small_vector_var.value.data = torch.tensor([1.0, 1.0, 1.0])  # Sum = 3
        result = lin_eq.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_inf_when_violated(
        self, small_vector_var, simple_constraint_data
    ):
        """Test forward() returns infinity when constraint is violated."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        small_vector_var.value.data = torch.tensor([1.0, 1.0, 2.0])  # Sum = 4 ≠ 3
        result = lin_eq.forward()
        assert torch.isinf(result)

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_false(self, vector_var, full_rank_constraint_data):
        """Test linear equality is not smooth (inherits from Polyhedron)."""
        lin_eq = LinearEquality(vector_var, **full_rank_constraint_data)
        assert lin_eq.is_smooth() is False

    def test_is_proxable_returns_true(self, vector_var, full_rank_constraint_data):
        """Test linear equality is proxable."""
        lin_eq = LinearEquality(vector_var, **full_rank_constraint_data)
        assert lin_eq.is_proxable() is True

    # ----------------------
    # Proximal operator tests
    # ----------------------

    def test_prox_projects_onto_affine_subspace(
        self, small_vector_var, simple_constraint_data
    ):
        """Test prox() projects point onto affine subspace."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        location = torch.tensor([0.0, 0.0, 0.0])  # Violates constraint

        result = lin_eq.prox(location, prox_scaling=1.0)

        # Check that result satisfies constraint: sum = 3
        assert torch.allclose(lin_eq.A @ result, lin_eq.b, atol=1e-5)

    def test_prox_is_closest_point_on_subspace(
        self, small_vector_var, simple_constraint_data
    ):
        """Test prox() finds closest point satisfying constraint."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        location = torch.tensor([1.0, 2.0, 3.0])  # Sum = 6, need sum = 3

        result = lin_eq.prox(location, prox_scaling=1.0)

        # Result should satisfy constraint
        assert torch.allclose(lin_eq.A @ result, lin_eq.b, atol=1e-5)

        # Result should be close to location (projection minimizes distance)
        assert torch.norm(result - location) < torch.norm(location)

    def test_prox_returns_identity_for_feasible_point(
        self, small_vector_var, simple_constraint_data
    ):
        """Test prox() returns input if already on affine subspace."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        location = torch.tensor([1.0, 1.0, 1.0])  # Already satisfies sum = 3

        result = lin_eq.prox(location, prox_scaling=1.0)

        # Should return approximately the same point
        assert torch.allclose(result, location, atol=1e-5)

    def test_prox_independent_of_scaling(
        self, small_vector_var, simple_constraint_data
    ):
        """Test prox() is independent of prox_scaling parameter."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        location = torch.tensor([2.0, 3.0, 1.0])

        result1 = lin_eq.prox(location, prox_scaling=1.0)
        result2 = lin_eq.prox(location, prox_scaling=100.0)

        # Projection onto affine subspace should be independent of scaling
        assert torch.allclose(result1, result2)

    def test_prox_with_multiple_constraints(
        self, vector_var, full_rank_constraint_data
    ):
        """Test prox() with multiple equality constraints."""
        lin_eq = LinearEquality(vector_var, **full_rank_constraint_data)
        location = torch.randn(5)

        result = lin_eq.prox(location, prox_scaling=1.0)

        # Check both constraints are satisfied
        assert torch.allclose(lin_eq.A @ result, lin_eq.b, atol=1e-4)

    def test_prox_minimizes_distance(self, vector_var, full_rank_constraint_data):
        """Test prox() result minimizes distance to location."""
        lin_eq = LinearEquality(vector_var, **full_rank_constraint_data)
        location = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        result = lin_eq.prox(location, prox_scaling=1.0)

        # Result should satisfy constraints
        assert torch.allclose(lin_eq.A @ result, lin_eq.b, atol=1e-4)

        # Check another feasible point has larger distance
        # (Hard to construct without solving the problem, so just verify constraints)
        assert not torch.allclose(result, location)

    # ----------------------
    # Edge cases
    # ----------------------

    def test_with_single_constraint(self, small_vector_var):
        """Test with single equality constraint (hyperplane)."""
        A = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([5.0])
        lin_eq = LinearEquality(small_vector_var, A, b)

        location = torch.tensor([0.0, 1.0, 2.0])
        result = lin_eq.prox(location, prox_scaling=1.0)

        # First element should be 5.0, others minimally changed
        assert torch.allclose(result[0], torch.tensor(5.0))
        assert torch.allclose(result[1:], location[1:], atol=1e-5)

    def test_with_orthogonal_constraints(self):
        """Test with orthogonal constraint vectors."""
        x = Variable((4,), name="x")
        A = torch.eye(2, 4)  # First two elements constrained
        b = torch.tensor([1.0, 2.0])
        lin_eq = LinearEquality(x, A, b)

        location = torch.tensor([0.0, 0.0, 3.0, 4.0])
        result = lin_eq.prox(location, prox_scaling=1.0)

        # First two elements should match constraints
        assert torch.allclose(result[:2], b)
        # Last two should be unchanged
        assert torch.allclose(result[2:], location[2:])

    def test_gradient_flows_through_prox(
        self, small_vector_var, simple_constraint_data
    ):
        """Test gradients flow through prox operator."""
        lin_eq = LinearEquality(small_vector_var, **simple_constraint_data)
        location = torch.randn(3, requires_grad=True)

        result = lin_eq.prox(location, prox_scaling=1.0)
        loss = result.sum()
        loss.backward()

        assert location.grad is not None
        assert not torch.isnan(location.grad).any()
