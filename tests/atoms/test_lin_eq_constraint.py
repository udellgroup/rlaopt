"""Tests for LinEqConstraint atom."""

import pytest
import torch

from rlaopt.atoms.lin_eq_constraint import LinEqConstraint
from rlaopt.expression import Variable


@pytest.fixture
def simple_constraint():
    """Fixture for a simple 2D constraint: x[0] + x[1] = 1."""
    x = Variable((2,), name="x")
    A = torch.tensor([[1.0, 1.0]])
    b = torch.tensor([1.0])
    return LinEqConstraint(x, A, b), A, b


@pytest.fixture
def multi_constraint():
    """Fixture for multiple equality constraints."""
    x = Variable((4,), name="x")
    A = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([2.0, 4.0])
    return LinEqConstraint(x, A, b), A, b


@pytest.fixture
def zero_constraint():
    """Fixture for linear subspace constraint: A @ x = 0."""
    x = Variable((3,), name="x")
    A = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([0.0])
    return LinEqConstraint(x, A, b), A, b


class TestLinEqConstraintInit:
    """Tests for LinEqConstraint initialization."""

    def test_single_constraint(self, simple_constraint):
        """Test initialization with a single equality constraint."""
        constraint, _, _ = simple_constraint

        assert constraint is not None
        assert constraint.is_proxable()

    def test_multiple_constraints(self, multi_constraint):
        """Test initialization with multiple equality constraints."""
        constraint, _, _ = multi_constraint

        assert constraint is not None
        assert constraint.is_proxable()

    @pytest.mark.parametrize(
        "n_vars,n_constraints",
        [
            (3, 1),
            (5, 3),
            (10, 5),
            (4, 4),  # Square system
        ],
    )
    def test_various_dimensions(self, n_vars, n_constraints):
        """Test initialization with various dimensions."""
        x = Variable((n_vars,), name="x")
        A = torch.randn(n_constraints, n_vars)
        b = torch.randn(n_constraints)

        constraint = LinEqConstraint(x, A, b)

        assert constraint is not None
        assert constraint.is_proxable()


class TestLinEqConstraintProx:
    """Tests for proximal operator (projection)."""

    def test_projection_satisfies_constraint(self, simple_constraint):
        """Test that projection satisfies the equality constraint."""
        constraint, A, b = simple_constraint

        location = torch.tensor([0.5, 0.5])
        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(A @ projected, b, atol=1e-6)

    def test_projection_already_feasible(self, simple_constraint):
        """Test projection of a point already on the affine subspace."""
        constraint, A, b = simple_constraint

        # Point already satisfies constraint: [0.3, 0.7] sums to 1
        location = torch.tensor([0.3, 0.7])
        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(projected, location, atol=1e-6)
        assert torch.allclose(A @ projected, b, atol=1e-6)

    @pytest.mark.parametrize(
        "location",
        [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([5.0, -3.0]),
            torch.tensor([-2.0, 8.0]),
        ],
    )
    def test_projection_various_points(self, simple_constraint, location):
        """Test projection from various starting points."""
        constraint, A, b = simple_constraint

        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(A @ projected, b, atol=1e-6)

    def test_projection_orthogonal(self):
        """Test that projection is orthogonal to the affine subspace."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 0.0, 0.0]])
        b = torch.tensor([2.0])
        constraint = LinEqConstraint(x, A, b)

        location = torch.tensor([5.0, 3.0, 4.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        # Projection should be [2.0, 3.0, 4.0] (only x[0] changes)
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(projected, expected, atol=1e-6)

    def test_projection_multiple_constraints(self, multi_constraint):
        """Test projection with multiple equality constraints."""
        constraint, A, b = multi_constraint

        location = torch.ones(4)
        projected = constraint.prox(location, prox_scaling=1.0)

        # Verify all constraints are satisfied
        assert torch.allclose(A @ projected, b, atol=1e-6)

    def test_projection_minimizes_distance(self):
        """Test that projection minimizes distance to original point."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([0.0])
        constraint = LinEqConstraint(x, A, b)

        location = torch.tensor([1.0, 2.0, 3.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        # Check constraint satisfaction
        assert torch.allclose(A @ projected, b, atol=1e-6)

        # Check that any other feasible point is farther
        alternative = torch.tensor([1.0, 1.0, -2.0])  # Also satisfies A @ x = 0
        dist_projected = torch.norm(projected - location)
        dist_alternative = torch.norm(alternative - location)
        assert dist_projected <= dist_alternative + 1e-6

    @pytest.mark.parametrize("prox_scaling", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_prox_scaling_independence(self, simple_constraint, prox_scaling):
        """Test that projection is independent of prox_scaling parameter."""
        constraint, A, b = simple_constraint

        location = torch.tensor([3.0, -1.0])

        proj_baseline = constraint.prox(location, prox_scaling=1.0)
        proj_scaled = constraint.prox(location, prox_scaling=prox_scaling)

        # Should be identical regardless of scaling
        assert torch.allclose(proj_baseline, proj_scaled, atol=1e-6)

    def test_projection_zero_rhs(self, zero_constraint):
        """Test constraint with zero right-hand side (linear subspace)."""
        constraint, A, b = zero_constraint

        location = torch.tensor([1.0, 1.0, 1.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(A @ projected, b, atol=1e-6)


class TestLinEqConstraintEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize(
        "n_vars,n_constraints",
        [
            (20, 5),
            (50, 10),
            (100, 20),
        ],
    )
    def test_large_scale(self, n_vars, n_constraints):
        """Test with larger constraint systems."""
        x = Variable((n_vars,), name="x")
        A = torch.randn(n_constraints, n_vars)
        b = torch.randn(n_constraints)
        constraint = LinEqConstraint(x, A, b)

        location = torch.randn(n_vars)
        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(A @ projected, b, atol=1e-5)

    def test_orthonormal_rows(self):
        """Test with orthonormal constraint matrix rows."""
        x = Variable((4,), name="x")
        # Create orthonormal rows via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(4, 4))
        A = Q[:2, :]  # Take first 2 rows
        b = torch.tensor([1.0, -1.0])
        constraint = LinEqConstraint(x, A, b)

        location = torch.randn(4)
        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(A @ projected, b, atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point precisions."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
        b = torch.tensor([1.0], dtype=dtype)
        constraint = LinEqConstraint(x, A, b)

        location = torch.randn(3, dtype=dtype)
        projected = constraint.prox(location, prox_scaling=1.0)

        atol = 1e-6 if dtype == torch.float32 else 1e-12
        assert torch.allclose(A @ projected, b, atol=atol)
        assert projected.dtype == dtype

    def test_very_small_values(self):
        """Test numerical stability with very small values."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([1e-10], dtype=torch.float64)
        constraint = LinEqConstraint(x, A, b)

        location = torch.tensor([1e-11, 2e-11, 3e-11], dtype=torch.float64)
        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(A @ projected, b, atol=1e-15)


class TestLinEqConstraintProperties:
    """Tests for constraint properties."""

    def test_is_proxable_always_true(self, simple_constraint):
        """Test that is_proxable always returns True."""
        constraint, _, _ = simple_constraint

        assert constraint.is_proxable() is True

    def test_idempotency(self, simple_constraint):
        """Test that projecting twice gives the same result."""
        constraint, _, _ = simple_constraint

        location = torch.randn(2)
        projected_once = constraint.prox(location, prox_scaling=1.0)
        projected_twice = constraint.prox(projected_once, prox_scaling=1.0)

        assert torch.allclose(projected_once, projected_twice, atol=1e-6)

    @pytest.mark.parametrize("seed", range(5))
    def test_idempotency_multiple_seeds(self, simple_constraint, seed):
        """Test idempotency with multiple random initializations."""
        constraint, _, _ = simple_constraint

        torch.manual_seed(seed)
        location = torch.randn(2)
        projected_once = constraint.prox(location, prox_scaling=1.0)
        projected_twice = constraint.prox(projected_once, prox_scaling=1.0)

        assert torch.allclose(projected_once, projected_twice, atol=1e-6)


class TestLinEqConstraintExamples:
    """Tests based on docstring examples."""

    def test_docstring_example_single(self):
        """Test the single equality constraint example from docstring."""
        x = Variable((2,), name="x")
        A = torch.tensor([[1.0, 1.0]])
        b = torch.tensor([1.0])
        constraint = LinEqConstraint(x, A, b)

        assert constraint is not None
        assert constraint.is_proxable()

    def test_docstring_example_multiple(self):
        """Test the multiple equality constraints example from docstring."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        constraint = LinEqConstraint(x, A, b)

        assert constraint is not None

    def test_docstring_example_projection(self):
        """Test the projection example from docstring."""
        torch.manual_seed(42)
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        constraint = LinEqConstraint(x, A, b)

        unconstrained_point = torch.randn(5)
        projected = constraint.prox(unconstrained_point, prox_scaling=1.0)

        # Verify: A @ projected should equal b
        assert torch.allclose(A @ projected, b, atol=1e-6)


@pytest.mark.parametrize(
    "n_vars,n_constraints,seed",
    [
        (3, 1, 0),
        (5, 2, 1),
        (10, 5, 2),
        (20, 10, 3),
    ],
)
def test_projection_general_case(n_vars, n_constraints, seed):
    """General test for projection with various configurations."""
    torch.manual_seed(seed)

    x = Variable((n_vars,), name="x")
    A = torch.randn(n_constraints, n_vars)
    b = torch.randn(n_constraints)
    constraint = LinEqConstraint(x, A, b)

    location = torch.randn(n_vars)
    projected = constraint.prox(location, prox_scaling=1.0)

    # Verify constraint satisfaction
    assert torch.allclose(A @ projected, b, atol=1e-5)

    # Verify idempotency
    projected_twice = constraint.prox(projected, prox_scaling=1.0)
    assert torch.allclose(projected, projected_twice, atol=1e-6)
