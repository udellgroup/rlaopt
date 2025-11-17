"""Tests for Polyhedron atom."""

import pytest
import torch

from rlaopt.atoms.polyhedron import Polyhedron
from rlaopt.expression import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="x")


@pytest.fixture
def full_constraint_data():
    """Create full constraint data with both equality and inequality."""
    return {
        "A": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 0.0, 1.0, 0.0]]),
        "b": torch.tensor([15.0, 2.0]),
        "C": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
        "lower": torch.tensor([-1.0, -2.0]),
        "upper": torch.tensor([5.0, 3.0]),
    }


@pytest.fixture
def inequality_only_data():
    """Create inequality constraint data only."""
    return {"C": torch.eye(5), "lower": torch.zeros(5), "upper": torch.ones(5)}


class TestPolyhedron:
    """Test Polyhedron constraint atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_full_constraints(self, vector_var, full_constraint_data):
        """Test initialization with both equality and inequality constraints."""
        poly = Polyhedron(vector_var, **full_constraint_data)
        assert poly.get_buffer("A") is not None
        assert poly.get_buffer("b") is not None
        assert poly.get_buffer("C") is not None
        assert poly.get_buffer("lower") is not None
        assert poly.get_buffer("upper") is not None

    def test_init_with_inequality_only(self, vector_var, inequality_only_data):
        """Test initialization with only inequality constraints."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        assert poly.get_buffer("A") is None
        assert poly.get_buffer("b") is None
        assert poly.get_buffer("C") is not None

    def test_init_with_scalar_bounds(self, vector_var):
        """Test initialization with scalar lower and upper bounds."""
        poly = Polyhedron(vector_var, C=torch.eye(5), lower=0.0, upper=1.0)
        assert poly.get_buffer("lower").shape == torch.Size([])
        assert poly.get_buffer("upper").shape == torch.Size([])

    def test_init_auto_fills_missing_bounds(self, vector_var):
        """Test initialization auto-fills infinity for missing bounds."""
        # Only upper provided
        poly_upper = Polyhedron(vector_var, C=torch.eye(5), upper=torch.ones(5))
        assert (
            torch.isinf(poly_upper.get_buffer("lower"))
            and poly_upper.get_buffer("lower") < 0
        )

        # Only lower provided
        poly_lower = Polyhedron(vector_var, C=torch.eye(5), lower=torch.zeros(5))
        assert (
            torch.isinf(poly_lower.get_buffer("upper"))
            and poly_lower.get_buffer("upper") > 0
        )

    def test_init_raises_error_if_A_without_b(self, vector_var):
        """Test initialization raises ValueError if A provided without b."""
        A = torch.randn(2, 5)
        with pytest.raises(ValueError, match="b cannot be None when A is not None"):
            Polyhedron(vector_var, A=A)

    def test_init_raises_error_if_b_without_A(self, vector_var):
        """Test initialization raises ValueError if b provided without A."""
        b = torch.randn(2)
        with pytest.raises(ValueError, match="A cannot be None when b is not None"):
            Polyhedron(vector_var, b=b)

    def test_init_raises_error_with_no_constraints(self, vector_var):
        """Test initialization raises ValueError with no constraints."""
        with pytest.raises(ValueError, match="trivial polyhedron"):
            Polyhedron(vector_var)

    # ----------------------
    # Validation tests
    # ----------------------

    def test_validation_catches_dimension_mismatch_A_b(self, vector_var):
        """Test validation catches dimension mismatch between A and b."""
        A = torch.randn(2, 5)
        b = torch.randn(3)  # Wrong size
        with pytest.raises(ValueError, match="matching row counts"):
            Polyhedron(vector_var, A=A, b=b)

    def test_validation_catches_dimension_mismatch_C_bounds(self, vector_var):
        """Test validation catches dimension mismatch between C and bounds."""
        C = torch.randn(3, 5)
        lower = torch.zeros(2)  # Wrong size
        upper = torch.ones(3)
        with pytest.raises(ValueError, match="matching row counts"):
            Polyhedron(vector_var, C=C, lower=lower, upper=upper)

    def test_validation_catches_zero_matrix_A(self, vector_var):
        """Test validation catches zero constraint matrix A."""
        A = torch.zeros(2, 5)
        b = torch.randn(2)
        with pytest.raises(ValueError, match="A must be non-zero"):
            Polyhedron(vector_var, A=A, b=b)

    def test_validation_catches_zero_matrix_C(self, vector_var):
        """Test validation catches zero constraint matrix C."""
        C = torch.zeros(3, 5)
        lower = torch.zeros(3)
        upper = torch.ones(3)
        with pytest.raises(ValueError, match="C must be non-zero"):
            Polyhedron(vector_var, C=C, lower=lower, upper=upper)

    def test_validation_catches_overdetermined_system(self, vector_var):
        """Test validation catches overdetermined equality constraints."""
        A = torch.randn(6, 5)  # More rows than columns
        b = torch.randn(6)
        with pytest.raises(ValueError, match="more columns than rows"):
            Polyhedron(vector_var, A=A, b=b)

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_zero_when_satisfied(
        self, vector_var, inequality_only_data
    ):
        """Test forward() returns 0 when constraints are satisfied."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        vector_var.value = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        result = poly.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_inf_when_violated(self, vector_var, inequality_only_data):
        """Test forward() returns infinity when constraints are violated."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        vector_var.value = torch.tensor(
            [2.0, 0.5, 0.5, 0.5, 0.5]
        )  # Violates upper bound
        result = poly.forward()
        assert torch.isinf(result)

    def test_forward_with_mixed_constraints(self, vector_var, full_constraint_data):
        """Test forward() with both equality and inequality constraints."""
        poly = Polyhedron(vector_var, **full_constraint_data)

        # Satisfying value: A @ x = b and lower <= C @ x <= upper
        vector_var.value = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

        result = poly.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_detects_equality_violation(self, vector_var, full_constraint_data):
        """Test forward() detects equality constraint violation."""
        poly = Polyhedron(vector_var, **full_constraint_data)
        vector_var.value = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0]
        )  # Doesn't satisfy A @ x = b
        result = poly.forward()
        assert torch.isinf(result)

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_false(self, vector_var, inequality_only_data):
        """Test polyhedron is not smooth."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        assert poly.is_smooth() is False

    def test_is_proxable_returns_false(self, vector_var, inequality_only_data):
        """Test polyhedron is not proxable."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        assert poly.is_proxable() is False

    def test_is_subsamplable_returns_false(self, vector_var, inequality_only_data):
        """Test polyhedron is not subsamplable."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        assert poly.is_subsamplable() is False

    def test_subsample_raises_not_implemented(self, vector_var, inequality_only_data):
        """Test subsample() raises NotImplementedError."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        with pytest.raises(NotImplementedError, match="not subsamplable"):
            poly.subsample(torch.tensor([0, 1, 2]))

    # ----------------------
    # Edge cases
    # ----------------------

    def test_constraints_at_boundary(self, vector_var, inequality_only_data):
        """Test constraints satisfied exactly at boundary."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        vector_var.value = torch.ones(5)  # Exactly at upper bound
        result = poly.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_with_unbounded_constraints(self, vector_var):
        """Test with only lower bounds (upper = infinity)."""
        poly = Polyhedron(vector_var, C=torch.eye(5), lower=torch.zeros(5))
        vector_var.value = torch.tensor(
            [100.0, 200.0, 300.0, 400.0, 500.0]
        )  # Very large values
        result = poly.forward()
        assert torch.allclose(result, torch.tensor(0.0))  # Should satisfy lower >= 0


class TestPolyhedronScaling:
    """Test Polyhedron scalar multiplication behavior."""

    def test_scaling_returns_same_atom(self, vector_var, inequality_only_data):
        """Test that scaling a Polyhedron returns the same atom (identity)."""
        poly = Polyhedron(vector_var, **inequality_only_data)
        scaled_poly = 5.0 * poly

        # Should return the exact same instance
        assert scaled_poly is poly

    def test_scaling_by_zero_warns(self, vector_var, inequality_only_data):
        """Test that scaling by zero raises a warning."""
        poly = Polyhedron(vector_var, **inequality_only_data)

        with pytest.warns(UserWarning, match="Scaling a Polyhedron.*by zero"):
            scaled_poly = 0.0 * poly

        # Should still return the same instance
        assert scaled_poly is poly
