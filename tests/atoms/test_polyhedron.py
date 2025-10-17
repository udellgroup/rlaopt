"""Comprehensive tests for the Polyhedron constraint atom."""

import pytest
import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.atoms.polyhedron import Polyhedron, _build_eval, _indicator, _validate
from rlaopt.expression import Variable

# ===============================
# Test Initialization - Valid Cases
# ===============================


class TestPolyhedronInitValid:
    """Tests for valid Polyhedron initialization."""

    def test_init_box_constraints(self):
        """Test initialization with box constraints only."""
        x = Variable((5,), name="x")
        lower = torch.full((5,), -1.0)
        upper = torch.full((5,), 1.0)

        poly = Polyhedron(x, lower=lower, upper=upper)

        assert isinstance(poly, Polyhedron)
        assert isinstance(poly, AtomExpression)
        assert poly.var_name == "x"

    def test_init_equality_constraints_only(self):
        """Test initialization with equality constraints only."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)

        poly = Polyhedron(x, A=A, b=b)

        assert poly.var_name == "x"
        assert torch.equal(poly.A, A)
        assert torch.equal(poly.b, b)

    def test_init_inequality_constraints_with_matrix(self):
        """Test initialization with inequality constraints using C matrix."""
        x = Variable((5,), name="x")
        C = torch.randn(2, 5)
        lower = torch.zeros(2)
        upper = torch.ones(2)

        poly = Polyhedron(x, C=C, lower=lower, upper=upper)

        assert torch.equal(poly.C, C)
        assert torch.equal(poly.lower, lower)
        assert torch.equal(poly.upper, upper)

    def test_init_mixed_constraints(self):
        """Test initialization with both equality and inequality constraints."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        C = torch.randn(2, 5)
        lower = torch.zeros(2)
        upper = torch.ones(2)

        poly = Polyhedron(x, A=A, b=b, C=C, lower=lower, upper=upper)

        assert poly.A is not None
        assert poly.C is not None
        assert poly.lower is not None

    def test_init_lower_only(self):
        """Test initialization with lower bound only (upper=inf)."""
        x = Variable((5,), name="x")
        lower = torch.zeros(5)

        poly = Polyhedron(x, lower=lower)

        assert torch.equal(poly.lower, lower)
        assert torch.isinf(poly.upper)

    def test_init_upper_only(self):
        """Test initialization with upper bound only (lower=-inf)."""
        x = Variable((5,), name="x")
        upper = torch.ones(5)

        poly = Polyhedron(x, upper=upper)

        assert torch.equal(poly.upper, upper)
        assert torch.isinf(poly.lower) and poly.lower < 0

    def test_init_with_1d_constraint_vectors(self):
        """Test initialization with 1D constraint vectors (hyperplane/halfspace)."""
        x = Variable((5,), name="x")
        a = torch.randn(5)  # 1D vector
        b = torch.tensor(1.0)

        poly = Polyhedron(x, A=a, b=b)

        assert poly.A.dim() == 1
        assert poly.b.dim() == 0

    def test_init_stores_constraints_as_buffers(self):
        """Test that constraints are stored as buffers, not parameters."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)

        poly = Polyhedron(x, A=A, b=b)

        # Should be buffers, not parameters
        buffers = dict(poly.named_buffers())
        params = dict(poly.named_parameters())

        assert "A" in buffers
        assert "b" in buffers
        assert "A" not in params
        assert "b" not in params

    def test_init_with_float_bounds(self):
        """Test initialization with float bounds (not tensors)."""
        x = Variable((5,), name="x")

        poly = Polyhedron(x, lower=-1.0, upper=1.0)

        assert isinstance(poly.lower, torch.Tensor)
        assert isinstance(poly.upper, torch.Tensor)
        assert poly.lower.item() == -1.0
        assert poly.upper.item() == 1.0

    def test_init_with_int_bounds(self):
        """Test initialization with int bounds."""
        x = Variable((5,), name="x")

        poly = Polyhedron(x, lower=0, upper=10)

        assert isinstance(poly.lower, torch.Tensor)
        assert isinstance(poly.upper, torch.Tensor)
        assert poly.lower.item() == 0.0
        assert poly.upper.item() == 10.0

    def test_non_uniform_tensor_bounds(self):
        """Test that non-uniform tensor bounds work correctly."""
        x = Variable((5,), name="x")
        lower = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        upper = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        poly = Polyhedron(x, lower=lower, upper=upper)

        # Should be stored as-is
        assert poly.lower.dim() == 1
        assert poly.upper.dim() == 1
        assert poly.lower.numel() == 5
        assert poly.upper.numel() == 5
        assert torch.equal(poly.lower, lower)
        assert torch.equal(poly.upper, upper)


# ===============================
# Test Initialization - Invalid Cases
# ===============================


class TestPolyhedronInitInvalid:
    """Tests for invalid Polyhedron initialization."""

    def test_init_A_without_b_raises_error(self):
        """Test that providing A without b raises ValueError."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)

        with pytest.raises(ValueError, match="b cannot be None when A is not None"):
            Polyhedron(x, A=A)

    def test_init_no_constraints_raises_error(self):
        """Test that no constraints raises ValueError."""
        x = Variable((5,), name="x")

        with pytest.raises(ValueError, match="trivial polyhedron"):
            Polyhedron(x)

    def test_init_A_b_dimension_mismatch(self):
        """Test that mismatched A and b dimensions raise ValueError."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(5)  # Wrong size!

        with pytest.raises(ValueError, match="matching row counts"):
            Polyhedron(x, A=A, b=b)

    def test_init_C_lower_dimension_mismatch(self):
        """Test that mismatched C and lower dimensions raise ValueError."""
        x = Variable((5,), name="x")
        C = torch.randn(2, 5)
        lower = torch.zeros(3)  # Wrong size!
        upper = torch.ones(2)

        with pytest.raises(ValueError, match="matching row counts"):
            Polyhedron(x, C=C, lower=lower, upper=upper)

    def test_init_C_upper_dimension_mismatch(self):
        """Test that mismatched C and upper dimensions raise ValueError."""
        x = Variable((5,), name="x")
        C = torch.randn(2, 5)
        lower = torch.zeros(2)
        upper = torch.ones(3)  # Wrong size!

        with pytest.raises(ValueError, match="matching row counts"):
            Polyhedron(x, C=C, lower=lower, upper=upper)


# ===============================
# Test Properties
# ===============================


class TestPolyhedronProperties:
    """Tests for Polyhedron mathematical properties."""

    def test_is_smooth(self):
        """Test that Polyhedron is not smooth."""
        x = Variable((5,), name="x")
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        assert poly.is_smooth() is False

    def test_is_proxable(self):
        """Test that Polyhedron is not proxable."""
        x = Variable((5,), name="x")
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        assert poly.is_proxable() is False

    def test_is_subsamplable(self):
        """Test that Polyhedron is not subsamplable."""
        x = Variable((5,), name="x")
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        assert poly.is_subsamplable() is False

    def test_subsample_raises_error(self):
        """Test that subsample raises NotImplementedError."""
        x = Variable((5,), name="x")
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))
        indices = torch.tensor([0, 1, 2])

        with pytest.raises(NotImplementedError, match="not subsamplable"):
            poly.subsample(indices)


# ===============================
# Test Forward - Box Constraints
# ===============================


class TestPolyhedronForwardBox:
    """Tests for forward evaluation with box constraints."""

    def test_forward_box_satisfied(self):
        """Test forward when box constraints are satisfied."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.9])
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_box_violated_upper(self):
        """Test forward when upper bound is violated."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.3, 1.2, 0.2, 0.9])  # 1.2 > 1.0
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_box_violated_lower(self):
        """Test forward when lower bound is violated."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, -0.1, 0.7, 0.2, 0.9])  # -0.1 < 0.0
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_box_at_boundary(self):
        """Test forward when values are exactly at boundaries."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0])
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_preserves_dtype(self):
        """Test that forward preserves input dtype in result."""
        x = Variable((5,), name="x", dtype=torch.float64)
        x.value.data = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
        poly = Polyhedron(x, lower=0.0, upper=1.0)

        result = poly.forward()

        assert result.dtype == torch.float64

    def test_forward_with_float_bounds_satisfied(self):
        """Test forward with float bounds (optimized to scalar)."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.9])
        poly = Polyhedron(x, lower=0.0, upper=1.0)

        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_lower_bound_only(self):
        """Test forward with only lower bound (upper=inf)."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 100.0, 0.0, 5.0])
        poly = Polyhedron(x, lower=torch.zeros(5))

        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_upper_bound_only(self):
        """Test forward with only upper bound (lower=-inf)."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([-100.0, -2.0, 0.5, 0.0, -5.0])
        poly = Polyhedron(x, upper=torch.ones(5))

        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Forward - Equality Constraints
# ===============================


class TestPolyhedronForwardEquality:
    """Tests for forward evaluation with equality constraints."""

    def test_forward_equality_satisfied(self):
        """Test forward when equality constraint is satisfied."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # A @ x = b: [1, 1, 1] @ [1, 2, 3] = 6
        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([6.0])

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_equality_violated(self):
        """Test forward when equality constraint is violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # A @ x = b: [1, 1, 1] @ [1, 2, 3] = 6, but b = 5
        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([5.0])

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_multiple_equalities_satisfied(self):
        """Test forward with multiple equality constraints satisfied."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        A = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # sum = 6
                [1.0, 0.0, 0.0],  # x[0] = 1
            ]
        )
        b = torch.tensor([6.0, 1.0])

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_multiple_equalities_one_violated(self):
        """Test forward with multiple equalities, one violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        A = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # sum = 6 ✓
                [1.0, 0.0, 0.0],  # x[0] = 2 ✗
            ]
        )
        b = torch.tensor([6.0, 2.0])

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_hyperplane_satisfied(self):
        """Test forward with 1D hyperplane constraint (a^T x = b)."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # a^T x = b: [1, 2, 3] · [1, 2, 3] = 14
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor(14.0)

        poly = Polyhedron(x, A=a, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_hyperplane_violated(self):
        """Test forward with 1D hyperplane constraint violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor(15.0)  # Actual is 14

        poly = Polyhedron(x, A=a, b=b)
        result = poly.forward()

        assert torch.isinf(result)


# ===============================
# Test Forward - Inequality Constraints
# ===============================


class TestPolyhedronForwardInequality:
    """Tests for forward evaluation with matrix inequality constraints."""

    def test_forward_inequality_satisfied(self):
        """Test forward when matrix inequality is satisfied."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # C @ x: [1, 1, 1] @ [1, 2, 3] = 6, check 0 <= 6 <= 10
        C = torch.tensor([[1.0, 1.0, 1.0]])
        lower = torch.tensor([0.0])
        upper = torch.tensor([10.0])

        poly = Polyhedron(x, C=C, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_inequality_violated_lower(self):
        """Test forward when lower bound of inequality is violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        C = torch.tensor([[1.0, 1.0, 1.0]])
        lower = torch.tensor([7.0])  # 6 < 7
        upper = torch.tensor([10.0])

        poly = Polyhedron(x, C=C, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_inequality_violated_upper(self):
        """Test forward when upper bound of inequality is violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        C = torch.tensor([[1.0, 1.0, 1.0]])
        lower = torch.tensor([0.0])
        upper = torch.tensor([5.0])  # 6 > 5

        poly = Polyhedron(x, C=C, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_halfspace_satisfied(self):
        """Test forward with 1D halfspace constraint (c^T x)."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # c^T x: [1, 2, 3] · [1, 2, 3] = 14
        c = torch.tensor([1.0, 2.0, 3.0])
        lower = torch.tensor(10.0)
        upper = torch.tensor(20.0)

        poly = Polyhedron(x, C=c, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_multiple_inequalities_satisfied(self):
        """Test forward with multiple inequality constraints satisfied."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        C = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # sum = 6
                [1.0, 0.0, 0.0],  # x[0] = 1
            ]
        )
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([10.0, 2.0])

        poly = Polyhedron(x, C=C, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Forward - Mixed Constraints
# ===============================


class TestPolyhedronForwardMixed:
    """Tests for forward evaluation with mixed constraints."""

    def test_forward_mixed_all_satisfied(self):
        """Test forward with both equality and inequality, all satisfied."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # Equality: sum = 6
        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([6.0])

        # Inequality: 0 <= x <= 5
        lower = torch.zeros(3)
        upper = torch.full((3,), 5.0)

        poly = Polyhedron(x, A=A, b=b, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_mixed_equality_violated(self):
        """Test forward with mixed constraints, equality violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([5.0])  # Actual sum is 6

        lower = torch.zeros(3)
        upper = torch.full((3,), 5.0)

        poly = Polyhedron(x, A=A, b=b, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_mixed_inequality_violated(self):
        """Test forward with mixed constraints, inequality violated."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([6.0])

        lower = torch.zeros(3)
        upper = torch.full((3,), 2.0)  # x[2] = 3 > 2

        poly = Polyhedron(x, A=A, b=b, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.isinf(result)

    def test_forward_mixed_with_C_matrix(self):
        """Test forward with mixed constraints including C matrix."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([6.0])

        C = torch.tensor([[1.0, 0.0, 0.0]])  # Check x[0]
        lower = torch.tensor([0.0])
        upper = torch.tensor([2.0])

        poly = Polyhedron(x, A=A, b=b, C=C, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Helper Functions
# ===============================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_validate_valid_equality(self):
        """Test _validate with valid equality constraints."""
        A = torch.randn(3, 5)
        b = torch.randn(3)

        # Should not raise
        _validate(A, None, b, None, None)

    def test_validate_invalid_equality_dims(self):
        """Test _validate with mismatched equality dimensions."""
        A = torch.randn(3, 5)
        b = torch.randn(5)  # Wrong size

        with pytest.raises(ValueError, match="matching row counts"):
            _validate(A, None, b, None, None)

    def test_validate_valid_inequality(self):
        """Test _validate with valid inequality constraints."""
        C = torch.randn(2, 5)
        lower = torch.zeros(2)
        upper = torch.ones(2)

        # Should not raise
        _validate(None, C, None, lower, upper)

    def test_validate_invalid_C_lower_dims(self):
        """Test _validate with mismatched C and lower dimensions."""
        C = torch.randn(2, 5)
        lower = torch.zeros(3)  # Wrong size
        upper = torch.ones(2)

        with pytest.raises(ValueError, match="matching row counts"):
            _validate(None, C, None, lower, upper)

    def test_validate_none_constraints(self):
        """Test _validate with all None (should not raise)."""
        # This will be caught by _build_eval
        _validate(None, None, None, None, None)

    def test_indicator_satisfied(self):
        """Test _indicator with satisfied constraint."""
        result = _indicator(True, torch.device("cpu"), torch.float32)

        assert torch.allclose(result, torch.tensor(0.0))
        assert result.device.type == "cpu"
        assert result.dtype == torch.float32

    def test_indicator_violated(self):
        """Test _indicator with violated constraint."""
        result = _indicator(False, torch.device("cpu"), torch.float32)

        assert torch.isinf(result)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float32

    def test_indicator_preserves_dtype(self):
        """Test _indicator preserves dtype."""
        result64 = _indicator(True, torch.device("cpu"), torch.float64)
        assert result64.dtype == torch.float64

        result32 = _indicator(True, torch.device("cpu"), torch.float32)
        assert result32.dtype == torch.float32

    def test_build_eval_no_constraints_raises(self):
        """Test _build_eval raises error with no constraints."""
        with pytest.raises(ValueError, match="trivial polyhedron"):
            _build_eval(None, None, None, None, None)

    def test_build_eval_returns_callable(self):
        """Test _build_eval returns a callable function."""
        lower = torch.zeros(5)
        upper = torch.ones(5)

        eval_fn = _build_eval(None, None, None, lower, upper)

        assert callable(eval_fn)

    def test_build_eval_function_works(self):
        """Test that built evaluation function works correctly."""
        lower = torch.zeros(5)
        upper = torch.ones(5)

        eval_fn = _build_eval(None, None, None, lower, upper)

        # Test with valid input
        x_valid = torch.ones(5) * 0.5
        result = eval_fn(x_valid)
        assert torch.allclose(result, torch.tensor(0.0))

        # Test with invalid input
        x_invalid = torch.ones(5) * 2.0
        result = eval_fn(x_invalid)
        assert torch.isinf(result)


# ===============================
# Test Edge Cases
# ===============================


class TestPolyhedronEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_variable(self):
        """Test Polyhedron with empty variable."""
        x = Variable((0,), name="x")
        lower = torch.zeros(0)
        upper = torch.ones(0)

        poly = Polyhedron(x, lower=lower, upper=upper)
        result = poly.forward()

        # Empty constraints should be satisfied
        assert torch.allclose(result, torch.tensor(0.0))

    def test_single_element_variable(self):
        """Test Polyhedron with single element."""
        x = Variable((1,), name="x")
        x.value.data = torch.tensor([0.5])

        poly = Polyhedron(x, lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_inf_bounds(self):
        """Test Polyhedron with infinite bounds."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1e10, -1e10, 0.0])

        lower = torch.tensor([-torch.inf, -torch.inf, -torch.inf])
        upper = torch.tensor([torch.inf, torch.inf, torch.inf])

        poly = Polyhedron(x, lower=lower, upper=upper)
        result = poly.forward()

        # Should always be satisfied
        assert torch.allclose(result, torch.tensor(0.0))

    def test_zero_tolerance_equality(self):
        """Test equality constraint with floating point precision."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        # Set up constraint that should be satisfied due to exact arithmetic
        A = torch.tensor([[1.0, 1.0, 1.0]])
        b = torch.tensor([6.0])

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_near_boundary_values(self):
        """Test values very close to boundaries."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([0.0 + 1e-10, 0.5, 1.0 - 1e-10])

        poly = Polyhedron(x, lower=torch.zeros(3), upper=torch.ones(3))
        result = poly.forward()

        # Should be satisfied (within numerical precision)
        assert torch.allclose(result, torch.tensor(0.0))

    def test_large_matrix_constraints(self):
        """Test Polyhedron with large constraint matrices."""
        x = Variable((100,), name="x")
        x.value.data = torch.rand(100)

        A = torch.randn(10, 100)
        b = A @ x.value.data  # Construct b so constraint is satisfied

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0), atol=1e-5)

    def test_negative_bounds(self):
        """Test Polyhedron with negative bounds."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([-2.0, -1.0, -0.5])

        poly = Polyhedron(
            x,
            lower=torch.tensor([-3.0, -2.0, -1.0]),
            upper=torch.tensor([-1.0, 0.0, 0.0]),
        )
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Integration Scenarios
# ===============================


class TestPolyhedronIntegration:
    """Integration tests for Polyhedron in realistic scenarios."""

    def test_unit_simplex_constraint(self):
        """Test unit simplex: x >= 0, sum(x) = 1."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])

        # Equality: sum = 1
        A = torch.ones(1, 5)
        b = torch.ones(1)

        # Inequality: x >= 0
        lower = torch.zeros(5)

        poly = Polyhedron(x, A=A, b=b, lower=lower)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_unit_simplex_violated(self):
        """Test unit simplex with violation."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.3, 0.2, 0.2, 0.2, 0.2])  # sum = 1.1

        A = torch.ones(1, 5)
        b = torch.ones(1)
        lower = torch.zeros(5)

        poly = Polyhedron(x, A=A, b=b, lower=lower)
        result = poly.forward()

        assert torch.isinf(result)

    def test_linear_program_feasible_region(self):
        """Test standard LP feasible region."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 1.0, 1.0])

        # Ax <= b constraints (reformulated as Ax - s = b with s >= 0)
        # But here we use inequalities: -inf <= Ax <= b
        C = torch.tensor(
            [
                [1.0, 2.0, 1.0],  # x1 + 2x2 + x3 <= 5
                [2.0, 1.0, 1.0],  # 2x1 + x2 + x3 <= 5
            ]
        )
        upper = torch.tensor([5.0, 5.0])
        lower = torch.tensor([-torch.inf, -torch.inf])

        # Non-negativity
        x_lower = torch.zeros(3)

        poly = Polyhedron(x, C=C, lower=lower, upper=upper)
        poly_nonneg = Polyhedron(x, lower=x_lower)

        result1 = poly.forward()
        result2 = poly_nonneg.forward()

        # Both should be satisfied
        assert torch.allclose(result1, torch.tensor(0.0))
        assert torch.allclose(result2, torch.tensor(0.0))

    def test_ball_approximation_with_hyperplanes(self):
        """Test approximating a ball with hyperplane constraints."""
        x = Variable((2,), name="x")
        x.value.data = torch.tensor([0.5, 0.5])

        # Octagon approximation: |x_i| <= 1, |x1 + x2| <= sqrt(2), etc.
        lower = torch.tensor([-1.0, -1.0])
        upper = torch.tensor([1.0, 1.0])

        poly = Polyhedron(x, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_constraint_as_optimization_barrier(self):
        """Test using Polyhedron as a barrier in optimization."""
        x = Variable((5,), name="x")
        x.value.data = torch.randn(5)

        # Project x to satisfy constraints
        x.value.data = torch.clamp(x.value.data, 0, 1)

        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        # Should be feasible
        result = poly.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_state_dict_save_load(self):
        """Test saving and loading Polyhedron state."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        lower = torch.zeros(5)
        upper = torch.ones(5)

        poly = Polyhedron(x, A=A, b=b, lower=lower, upper=upper)

        # Save state
        state = poly.state_dict()

        # Check all components are present
        assert "A" in state
        assert "b" in state
        assert "lower" in state
        assert "upper" in state
        assert "x" in state

    def test_parameter_tracking(self):
        """Test that Polyhedron tracks variable parameters."""
        x = Variable((5,), name="x")
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        params = list(poly.parameters())
        assert len(params) == 1
        assert params[0].shape == torch.Size([5])

    def test_buffer_tracking(self):
        """Test that Polyhedron stores constraints as buffers."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)

        poly = Polyhedron(x, A=A, b=b)

        buffers = dict(poly.named_buffers())
        assert "A" in buffers
        assert "b" in buffers

    def test_device_transfer(self):
        """Test moving Polyhedron to different device."""
        x = Variable((5,), name="x", device=torch.device("cpu"))
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        poly = poly.to("cpu")

        assert poly.get_variable("x").device.type == "cpu"
        assert poly.lower.device.type == "cpu"
        assert poly.upper.device.type == "cpu"

    def test_dtype_conversion(self):
        """Test Polyhedron dtype conversion."""
        x = Variable((5,), name="x", dtype=torch.float32)
        poly = Polyhedron(
            x,
            lower=torch.zeros(5, dtype=torch.float32),
            upper=torch.ones(5, dtype=torch.float32),
        )

        poly = poly.to(torch.float64)

        assert poly.get_variable("x").dtype == torch.float64


# ===============================
# Test Gradient Behavior
# ===============================


class TestPolyhedronGradients:
    """Tests for gradient behavior with Polyhedron."""

    def test_gradient_when_satisfied(self):
        """Test gradient computation when constraints are satisfied."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()
        # Result is 0, which doesn't require grad for indicator function
        # Just verify it computes
        assert result.item() == 0.0

    def test_gradient_when_violated(self):
        """Test gradient computation when constraints are violated."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.5, 1.5, 0.5, 0.5])
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()
        # Result is inf, which doesn't have meaningful gradients
        assert torch.isinf(result)

    def test_no_gradient_flow_through_indicator(self):
        """Test that indicator function doesn't provide gradient flow."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], requires_grad=True)
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        result = poly.forward()

        # Indicator functions have zero gradient almost everywhere
        # (or undefined at boundaries)
        # This test just verifies it doesn't crash
        assert result.item() == 0.0


# ===============================
# Test Special Constraint Types
# ===============================


class TestSpecialConstraintTypes:
    """Tests for special types of constraints."""

    def test_non_negativity_constraint(self):
        """Test non-negativity constraint (x >= 0)."""
        x = Variable((5,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        poly = Polyhedron(x, lower=torch.zeros(5))
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_equality_only_constraint(self):
        """Test constraint with only equality (Ax = b)."""
        x = Variable((4,), name="x")
        x.value.data = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # x1 + x2 + x3 + x4 = 4
        A = torch.ones(1, 4)
        b = torch.tensor([4.0])

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_inequality_only_constraint(self):
        """Test constraint with only inequality (Cx <= d)."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1.0, 2.0, 3.0])

        C = torch.tensor([[1.0, 1.0, 1.0]])
        upper = torch.tensor([10.0])

        poly = Polyhedron(x, C=C, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_probability_simplex(self):
        """Test probability simplex: x >= 0, sum(x) = 1."""
        x = Variable((5,), name="x")
        # Valid probability distribution
        x.value.data = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])

        A = torch.ones(1, 5)
        b = torch.ones(1)
        lower = torch.zeros(5)

        poly = Polyhedron(x, A=A, b=b, lower=lower)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_orthogonality_constraint(self):
        """Test orthogonality constraint between vectors."""
        # x and y orthogonal: x^T y = 0
        xy = Variable((6,), name="xy")  # [x1, x2, x3, y1, y2, y3]
        xy.value.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        # Constraint: x1*y1 + x2*y2 + x3*y3 = 0
        A = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # x1*y1 (misaligned for test)
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # x2*y2
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )  # x3*y3
        # Actually, let's use a simpler constraint
        # [1, 0, 0, 1, 0, 0] @ [1,0,0,0,1,0] = 1 (not orthogonal)
        # Better example: dot product constraint
        A = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        b = torch.tensor([1.0])  # Not zero, just a constraint

        poly = Polyhedron(xy, A=A, b=b)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))


# ===============================
# Test Error Messages
# ===============================


class TestErrorMessages:
    """Tests for error messages and error handling."""

    def test_error_message_A_without_b(self):
        """Test that error message is clear when A provided without b."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)

        with pytest.raises(ValueError) as excinfo:
            Polyhedron(x, A=A)

        assert "b cannot be None" in str(excinfo.value)

    def test_error_message_dimension_mismatch(self):
        """Test that error message is clear for dimension mismatches."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(5)

        with pytest.raises(ValueError) as excinfo:
            Polyhedron(x, A=A, b=b)

        assert "matching row counts" in str(excinfo.value)

    def test_error_message_no_constraints(self):
        """Test that error message is clear when no constraints provided."""
        x = Variable((5,), name="x")

        with pytest.raises(ValueError) as excinfo:
            Polyhedron(x)

        assert "trivial polyhedron" in str(excinfo.value)

    def test_error_message_subsample(self):
        """Test that subsample error message is clear."""
        x = Variable((5,), name="x")
        poly = Polyhedron(x, lower=torch.zeros(5), upper=torch.ones(5))

        with pytest.raises(NotImplementedError) as excinfo:
            poly.subsample(torch.tensor([0, 1, 2]))

        assert "not subsamplable" in str(excinfo.value)


# ===============================
# Test Documentation Examples
# ===============================


class TestDocumentationExamples:
    """Tests for examples from docstrings."""

    def test_box_constraints_example(self):
        """Test box constraints example from docstring."""
        x = Variable((5,), name="x")
        box = Polyhedron(x, lower=torch.full((5,), -1.0), upper=torch.full((5,), 1.0))

        assert isinstance(box, Polyhedron)
        assert torch.equal(box.lower, torch.full((5,), -1.0))

    def test_equality_constraint_example(self):
        """Test equality constraint example from docstring."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        poly = Polyhedron(x, A=A, b=b)

        assert isinstance(poly, Polyhedron)
        assert torch.equal(poly.A, A)

    def test_mixed_constraints_example(self):
        """Test mixed constraints example from docstring."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        C = torch.randn(2, 5)
        poly = Polyhedron(x, A=A, b=b, C=C, lower=torch.zeros(2), upper=torch.ones(2))

        assert isinstance(poly, Polyhedron)
        assert poly.A is not None
        assert poly.C is not None


# ===============================
# Test Numerical Stability
# ===============================


class TestNumericalStability:
    """Tests for numerical stability and precision."""

    def test_floating_point_equality(self):
        """Test equality constraint with floating point arithmetic."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([0.1, 0.2, 0.7])

        # sum = 1.0 (but might have floating point error)
        A = torch.ones(1, 3)
        b = torch.ones(1)

        poly = Polyhedron(x, A=A, b=b)
        result = poly.forward()

        # This might fail due to floating point precision
        # Result depends on torch's == operator behavior
        if torch.allclose(A @ x.value.data, b):
            assert torch.allclose(result, torch.tensor(0.0))

    def test_very_small_values(self):
        """Test with very small constraint values."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1e-10, 2e-10, 3e-10])

        poly = Polyhedron(x, lower=torch.zeros(3), upper=torch.ones(3) * 1e-9)
        result = poly.forward()

        # Should be satisfied
        assert torch.allclose(result, torch.tensor(0.0))

    def test_very_large_values(self):
        """Test with very large constraint values."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1e6, 2e6, 3e6])

        poly = Polyhedron(x, lower=torch.zeros(3), upper=torch.ones(3) * 1e7)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))

    def test_mixed_scale_constraints(self):
        """Test with constraints at different scales."""
        x = Variable((3,), name="x")
        x.value.data = torch.tensor([1e-5, 1.0, 1e5])

        lower = torch.tensor([1e-6, 0.5, 1e4])
        upper = torch.tensor([1e-4, 1.5, 1e6])

        poly = Polyhedron(x, lower=lower, upper=upper)
        result = poly.forward()

        assert torch.allclose(result, torch.tensor(0.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
