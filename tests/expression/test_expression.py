"""
Pytest suite for Expression class and its composition mechanics.

Tests cover:
- Operator overloading (+, -, *, @, /, **)
- Expression composition and tree building
- Parameter tracking and updates
- Evaluation with different parameters
- Smoothness and proxability properties
- CVXPY conversion (if keeping this feature)
"""

import pytest
import torch
import numpy as np

from rlaopt.expression.expression import (
    Variable,
    ConstExpression,
    AddExpression,
    ProductExpression,
    UnaryOpExpression,
    to_expr,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def simple_variables():
    """Create simple test variables."""
    x = Variable(5, name="x")
    y = Variable(5, name="y")
    return x, y


@pytest.fixture
def matrix_variables():
    """Create matrix variables for testing matmul."""
    A = Variable(3, 4, name="A")
    b = Variable(4, name="b")
    return A, b


# ============================================================================
# Variable Tests
# ============================================================================


class TestVariable:
    """Test suite for Variable class."""

    def test_initialization_from_size(self):
        """Test creating variable from size."""
        x = Variable(5, name="x")
        assert x.value.shape == (5,)
        assert x.name == "x"
        assert isinstance(x.value, torch.nn.Parameter)
        assert x.value.requires_grad

    def test_initialization_from_shape(self):
        """Test creating variable from tuple shape."""
        x = Variable(3, 4, name="x")
        assert x.value.shape == (3, 4)

        # Also test tuple shape
        y = Variable((3, 4), name="y")
        assert y.value.shape == (3, 4)

    def test_initialization_from_tensor(self):
        """Test creating variable from existing tensor."""
        data = torch.randn(5)
        x = Variable(data, name="x")
        assert x.value.shape == (5,)
        assert torch.equal(x.value.data, data)

    def test_unique_ids(self):
        """Test that each variable gets unique ID."""
        x = Variable(5)
        y = Variable(5)
        assert x.id != y.id

    def test_custom_id(self):
        """Test creating variable with custom ID."""
        x = Variable(5, var_id=42, name="x")
        assert x.id == 42

    def test_default_name(self):
        """Test that default name is generated."""
        x = Variable(5)
        assert x.name.startswith("var")

    def test_is_smooth(self):
        """Variables are smooth."""
        x = Variable(5)
        assert x.is_smooth()

    def test_is_not_proxable(self):
        """Variables are not proxable."""
        x = Variable(5)
        assert not x.is_proxable()

    def test_forward(self):
        """Test forward returns the parameter value."""
        x = Variable(5)
        x.value.data = torch.ones(5)
        output = x.forward()
        assert torch.equal(output, torch.ones(5))

    def test_transpose(self):
        """Test transpose operation."""
        A = Variable(3, 4)
        A_T = A.transpose()

        A.value.data = torch.randn(3, 4)
        result = A_T.forward()
        assert result.shape == (4, 3)
        assert torch.equal(result, A.value.T)

    def test_transpose_property(self):
        """Test .T property."""
        A = Variable(3, 4)
        A_T = A.T

        A.value.data = torch.randn(3, 4)
        result = A_T.forward()
        assert result.shape == (4, 3)

    def test_transpose_1d(self):
        """Test that 1D transpose returns self."""
        x = Variable(5)
        x_T = x.transpose()
        assert x_T is x

    def test_sum(self):
        """Test sum operation."""
        x = Variable(3, 4)
        x.value.data = torch.ones(3, 4)

        # Sum all elements
        s = x.sum()
        assert s.forward().item() == pytest.approx(12.0)

        # Sum along dimension
        s_dim = x.sum(dim=0)
        result = s_dim.forward()
        assert result.shape == (4,)
        assert torch.allclose(result, torch.tensor([3.0, 3.0, 3.0, 3.0]))


# ============================================================================
# ConstExpression Tests
# ============================================================================


class TestConstExpression:
    """Test suite for ConstExpression class."""

    def test_initialization_from_float(self):
        """Test creating constant from float."""
        c = ConstExpression(3.14)
        assert c.value.item() == pytest.approx(3.14, rel=1e-6)
        assert c.value.dtype == torch.get_default_dtype()

    def test_initialization_from_int(self):
        """Test creating constant from int."""
        c = ConstExpression(5)
        assert c.value.item() == 5.0

    def test_initialization_from_tensor(self):
        """Test creating constant from tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        c = ConstExpression(t)
        assert torch.equal(c.value, t)

    def test_is_smooth(self):
        """Constants are smooth."""
        c = ConstExpression(5.0)
        assert c.is_smooth()

    def test_is_proxable(self):
        """Constants are proxable."""
        c = ConstExpression(5.0)
        assert c.is_proxable()

    def test_forward(self):
        """Test forward returns the constant value."""
        c = ConstExpression(5.0)
        assert c.forward().item() == pytest.approx(5.0)

    def test_negation(self):
        """Test that negating constant keeps it as constant."""
        c = ConstExpression(5.0)
        neg_c = -c
        assert isinstance(neg_c, ConstExpression)
        assert neg_c.forward().item() == pytest.approx(-5.0)

    def test_no_gradients(self):
        """Test that constants are buffers, not parameters."""
        c = ConstExpression(5.0)
        assert len(list(c.parameters())) == 0
        assert len(list(c.buffers())) == 1


# ============================================================================
# to_expr Helper Tests
# ============================================================================


class TestToExpr:
    """Test the to_expr helper function."""

    def test_expression_passthrough(self):
        """Expression should pass through unchanged."""
        x = Variable(5)
        result = to_expr(x)
        assert result is x

    def test_float_to_const(self):
        """Float should become ConstExpression."""
        result = to_expr(3.14)
        assert isinstance(result, ConstExpression)
        assert result.value.item() == pytest.approx(3.14)

    def test_int_to_const(self):
        """Int should become ConstExpression."""
        result = to_expr(5)
        assert isinstance(result, ConstExpression)
        assert result.value.item() == 5.0

    def test_tensor_to_const(self):
        """Tensor should become ConstExpression."""
        t = torch.tensor([1.0, 2.0])
        result = to_expr(t)
        assert isinstance(result, ConstExpression)
        assert torch.equal(result.value, t)

    def test_invalid_type_raises(self):
        """Invalid types should raise TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            to_expr("string")


# ============================================================================
# Addition Tests
# ============================================================================


class TestAddition:
    """Test suite for addition operations."""

    def test_add_two_variables(self, simple_variables):
        """Test x + y."""
        x, y = simple_variables
        z = x + y

        assert isinstance(z, AddExpression)
        assert len(z.exprs) == 2

        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 3)

    def test_add_variable_and_constant(self, simple_variables):
        """Test x + 5."""
        x, _ = simple_variables
        z = x + 5

        assert isinstance(z, AddExpression)

        x.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 7)

    def test_radd_constant_and_variable(self, simple_variables):
        """Test 5 + x."""
        x, _ = simple_variables
        z = 5 + x

        assert isinstance(z, AddExpression)

        x.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 7)

    def test_add_three_terms(self, simple_variables):
        """Test x + y + z."""
        x, y = simple_variables
        z = Variable(5, name="z")
        expr = x + y + z

        assert isinstance(expr, AddExpression)

        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2
        z.value.data = torch.ones(5) * 3
        result = expr.forward()
        assert torch.equal(result, torch.ones(5) * 6)

    def test_add_is_smooth_if_all_smooth(self, simple_variables):
        """Addition is smooth if all terms are smooth."""
        x, y = simple_variables
        z = x + y
        assert z.is_smooth()

    def test_subtraction(self, simple_variables):
        """Test x - y."""
        x, y = simple_variables
        z = x - y

        x.value.data = torch.ones(5) * 5
        y.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 3)

    def test_rsub(self, simple_variables):
        """Test 5 - x."""
        x, _ = simple_variables
        z = 5 - x

        x.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 3)

    def test_negation(self, simple_variables):
        """Test -x."""
        x, _ = simple_variables
        neg_x = -x

        x.value.data = torch.ones(5) * 3
        result = neg_x.forward()
        assert torch.equal(result, torch.ones(5) * -3)

    def test_mixed_shapes_scalar_vector(self, simple_variables):
        """Test addition with mixed shapes (scalar + vector)."""
        x, _ = simple_variables

        # scalar + vector
        expr = 10 + x
        x.value.data = torch.ones(5) * 2
        result = expr.forward()
        assert torch.equal(result, torch.ones(5) * 12)

        # vector + scalar + vector
        y = Variable(5, name="y")
        y.value.data = torch.ones(5) * 3
        expr2 = x + 5 + y
        result2 = expr2.forward()
        assert torch.equal(result2, torch.ones(5) * 10)

    def test_fast_path_same_shapes(self, simple_variables):
        """Test that same-shape addition uses fast path (stack+sum)."""
        x, y = simple_variables
        z = Variable(5, name="z")

        # All same shape - should use fast path
        expr = x + y + z

        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2
        z.value.data = torch.ones(5) * 3

        result = expr.forward()
        expected = torch.ones(5) * 6
        assert torch.equal(result, expected)

    def test_many_terms_same_shape(self):
        """Test adding many terms with same shape (performance test)."""
        # Create 20 variables
        variables = [Variable(100, name=f"x{i}") for i in range(20)]
        for i, var in enumerate(variables):
            var.value.data = torch.ones(100) * (i + 1)

        # Sum them all
        expr = sum(variables[1:], variables[0])
        result = expr.forward()

        # Should equal sum of 1+2+...+20 = 210
        expected = torch.ones(100) * 210
        assert torch.equal(result, expected)


# ============================================================================
# Multiplication Tests
# ============================================================================


class TestMultiplication:
    """Test suite for multiplication operations."""

    def test_mul_variable_and_constant(self, simple_variables):
        """Test x * 5 (elementwise)."""
        x, _ = simple_variables
        z = x * 5

        assert isinstance(z, ProductExpression)
        assert not z.matmul

        x.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 10)

    def test_rmul_constant_and_variable(self, simple_variables):
        """Test 5 * x."""
        x, _ = simple_variables
        z = 5 * x

        x.value.data = torch.ones(5) * 2
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 10)

    def test_mul_two_variables(self, simple_variables):
        """Test x * y (elementwise)."""
        x, y = simple_variables
        z = x * y

        x.value.data = torch.ones(5) * 2
        y.value.data = torch.ones(5) * 3
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 6)

    def test_division_by_scalar(self, simple_variables):
        """Test x / 2."""
        x, _ = simple_variables
        z = x / 2

        x.value.data = torch.ones(5) * 10
        result = z.forward()
        assert torch.equal(result, torch.ones(5) * 5)

    def test_division_by_variable_not_implemented(self, simple_variables):
        """Test x / y raises NotImplemented."""
        x, y = simple_variables
        with pytest.raises(TypeError):
            z = x / y

    def test_product_is_smooth(self, simple_variables):
        """Products are always smooth."""
        x, y = simple_variables
        z = x * y
        assert z.is_smooth()

    def test_product_not_proxable(self, simple_variables):
        """Products are not proxable."""
        x, y = simple_variables
        z = x * y
        assert not z.is_proxable()


# ============================================================================
# Matrix Multiplication Tests
# ============================================================================


class TestMatrixMultiplication:
    """Test suite for matrix multiplication (@)."""

    def test_matmul_matrix_vector(self, matrix_variables):
        """Test A @ b."""
        A, b = matrix_variables
        z = A @ b

        assert isinstance(z, ProductExpression)
        assert z.matmul

        A.value.data = torch.ones(3, 4) * 2
        b.value.data = torch.ones(4)
        result = z.forward()
        assert result.shape == (3,)
        assert torch.equal(result, torch.ones(3) * 8)

    def test_rmatmul_vector_matrix(self):
        """Test b @ A (as row vector)."""
        b = Variable(3, name="b")
        A = Variable(3, 4, name="A")
        z = b @ A

        # This should work if b is treated as row vector
        b.value.data = torch.ones(3)
        A.value.data = torch.ones(3, 4) * 2
        result = z.forward()
        assert result.shape == (4,)
        assert torch.equal(result, torch.ones(4) * 6)

    def test_matmul_composition(self):
        """Test A @ B @ c."""
        A = Variable(2, 3, name="A")
        B = Variable(3, 4, name="B")
        c = Variable(4, name="c")

        z = A @ B @ c

        A.value.data = torch.ones(2, 3)
        B.value.data = torch.ones(3, 4)
        c.value.data = torch.ones(4)

        result = z.forward()
        assert result.shape == (2,)
        assert torch.equal(result, torch.ones(2) * 12)


# ============================================================================
# Power Tests
# ============================================================================


class TestPower:
    """Test suite for power operations."""

    def test_square(self, simple_variables):
        """Test x ** 2."""
        x, _ = simple_variables
        x_sq = x**2

        assert isinstance(x_sq, UnaryOpExpression)

        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = x_sq.forward()
        expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
        assert torch.equal(result, expected)

    def test_cube(self, simple_variables):
        """Test x ** 3."""
        x, _ = simple_variables
        x_cube = x**3

        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = x_cube.forward()
        expected = torch.tensor([1.0, 8.0, 27.0, 64.0, 125.0])
        assert torch.equal(result, expected)


# ============================================================================
# Parameter Tracking Tests
# ============================================================================


class TestParameterTracking:
    """Test parameter tracking in composed expressions."""

    def test_single_variable_params(self, simple_variables):
        """Single variable should have one parameter."""
        x, _ = simple_variables
        params = list(x.parameters())
        assert len(params) == 1
        assert params[0] is x.value

    def test_sum_tracks_all_params(self, simple_variables):
        """Sum should track all constituent parameters."""
        x, y = simple_variables
        z = x + y

        params = dict(z.named_parameters())
        assert len(params) == 2
        # With Option 1, names should contain variable names
        param_keys = "".join(params.keys())
        assert x.name in param_keys
        assert y.name in param_keys

    def test_constant_has_no_params(self):
        """Constants should have no parameters."""
        c = ConstExpression(5.0)
        params = list(c.parameters())
        assert len(params) == 0

    def test_complex_expression_params(self, simple_variables):
        """Complex expression should track all variables."""
        x, y = simple_variables
        z = Variable(5, name="z")

        # expr = 2*x + y - 3*z
        expr = 2 * x + y - 3 * z

        params = dict(expr.named_parameters())
        assert len(params) == 3
        # Check that all variable names appear in the keys
        param_keys = "".join(params.keys())
        assert x.name in param_keys
        assert y.name in param_keys
        assert z.name in param_keys

    def test_params_dict_property(self, simple_variables):
        """Test params_dict() method."""
        x, y = simple_variables
        expr = x + y

        params = expr.params_dict()
        assert isinstance(params, dict)
        assert len(params) == 2
        # Variable names should appear in keys
        param_keys = "".join(params.keys())
        assert x.name in param_keys
        assert y.name in param_keys

    def test_params_property(self, simple_variables):
        """Test params property."""
        x, y = simple_variables
        expr = x + y

        params = expr.params
        assert isinstance(params, dict)
        assert len(params) == 2


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    """Test expression evaluation with different parameters."""

    def test_evaluate_at_different_params(self, simple_variables):
        """Test evaluating expression at different parameter values."""
        x, y = simple_variables
        expr = x + y

        # Set current values
        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2

        # Get actual parameter keys from state_dict
        state = expr.state_dict()
        keys = list(state.keys())
        assert len(keys) == 2

        # With Option 1, we can find keys by variable name
        x_key = [k for k in keys if x.name in k][0]
        y_key = [k for k in keys if y.name in k][0]

        # Evaluate at different params using actual keys
        new_params = {x_key: torch.ones(5) * 10, y_key: torch.ones(5) * 20}
        result = expr.evaluate(new_params)
        assert torch.equal(result, torch.ones(5) * 30)

        # Original values unchanged
        assert torch.equal(x.value, torch.ones(5))
        assert torch.equal(y.value, torch.ones(5) * 2)

    def test_update_params(self, simple_variables):
        """Test updating parameters from dictionary."""
        x, y = simple_variables
        expr = x + y

        # Get actual parameter names from state_dict
        state = expr.state_dict()
        keys = list(state.keys())

        # Find keys by variable name
        x_key = [k for k in keys if x.name in k][0]
        y_key = [k for k in keys if y.name in k][0]

        new_params = {x_key: torch.ones(5) * 5, y_key: torch.ones(5) * 10}
        expr.update_params(new_params)

        # Check that parameters were updated
        assert torch.equal(x.value, torch.ones(5) * 5)
        assert torch.equal(y.value, torch.ones(5) * 10)

    def test_forward_with_kwargs(self):
        """Test forward with keyword arguments."""
        # Create expression that accepts kwargs
        x = Variable(5, name="x")

        # Simulate an expression that uses kwargs
        # (This would be more relevant for atoms that take data)
        result = x.forward()
        assert result.shape == (5,)


# ============================================================================
# Composition and Tree Building Tests
# ============================================================================


class TestComposition:
    """Test complex expression composition."""

    def test_linear_combination(self):
        """Test a*x + b*y + c."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")

        expr = 2 * x + 3 * y + 5

        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2
        result = expr.forward()
        # 2*1 + 3*2 + 5 = 13
        assert torch.equal(result, torch.ones(5) * 13)

    def test_quadratic_form(self):
        """Test x^T @ A @ x (quadratic form)."""
        x = Variable(3, name="x")
        A = Variable(3, 3, name="A")

        # x @ A @ x
        expr = x @ A @ x

        x.value.data = torch.ones(3)
        A.value.data = torch.eye(3) * 2
        result = expr.forward()
        # [1,1,1] @ [[2,0,0],[0,2,0],[0,0,2]] @ [1,1,1] = 6
        assert result.item() == pytest.approx(6.0)

    def test_mixed_operations(self):
        """Test expression with multiple operation types."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")

        # (x + y)^2
        expr = (x + y) ** 2

        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2
        result = expr.forward()
        # (1 + 2)^2 = 9
        assert torch.equal(result, torch.ones(5) * 9)

    def test_nested_sums(self):
        """Test nested addition."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")
        z = Variable(5, name="z")

        # (x + y) + (y + z)
        expr1 = x + y
        expr2 = y + z
        expr = expr1 + expr2

        x.value.data = torch.ones(5)
        y.value.data = torch.ones(5) * 2
        z.value.data = torch.ones(5) * 3
        result = expr.forward()
        # 1 + 2 + 2 + 3 = 8
        assert torch.equal(result, torch.ones(5) * 8)


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test that gradients flow correctly through expressions."""

    def test_simple_gradient(self, simple_variables):
        """Test gradient through x + y."""
        x, y = simple_variables
        expr = x + y

        x.value.data = torch.ones(5, requires_grad=True)
        y.value.data = torch.ones(5, requires_grad=True) * 2

        result = expr.forward()
        loss = result.sum()
        loss.backward()

        # Both gradients should be 1
        assert torch.allclose(x.value.grad, torch.ones(5))
        assert torch.allclose(y.value.grad, torch.ones(5))

    def test_gradient_through_product(self, simple_variables):
        """Test gradient through x * y."""
        x, y = simple_variables
        expr = x * y

        x.value.data = torch.ones(5, requires_grad=True) * 2
        y.value.data = torch.ones(5, requires_grad=True) * 3

        result = expr.forward()
        loss = result.sum()
        loss.backward()

        # d/dx(x*y) = y, d/dy(x*y) = x
        assert torch.allclose(x.value.grad, torch.ones(5) * 3.0)
        assert torch.allclose(y.value.grad, torch.ones(5) * 2.0)

    def test_gradient_through_power(self, simple_variables):
        """Test gradient through x^2."""
        x, _ = simple_variables
        expr = x**2

        x.value.data = torch.ones(5, requires_grad=True) * 3

        result = expr.forward()
        loss = result.sum()
        loss.backward()

        # d/dx(x^2) = 2x
        assert torch.allclose(x.value.grad, torch.ones(5) * 6.0)


# ============================================================================
# UnaryOpExpression Tests
# ============================================================================


class TestUnaryOpExpression:
    """Test unary operations."""

    def test_custom_unary_op(self, simple_variables):
        """Test custom unary operation."""
        x, _ = simple_variables

        # Create custom op: abs(x)
        expr = UnaryOpExpression(x, torch.abs)

        x.value.data = torch.tensor([-1.0, 2.0, -3.0, 4.0, -5.0])
        result = expr.forward()
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.equal(result, expected)

    def test_unary_op_from_const(self):
        """Test unary op on constant."""
        c = ConstExpression(torch.tensor([-1.0, 2.0, -3.0]))
        expr = UnaryOpExpression(c, torch.abs)

        result = expr.forward()
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

    def test_unary_sum_method(self, simple_variables):
        """Test the sum() method of UnaryOpExpression."""
        x, _ = simple_variables
        x.value.data = torch.ones(5) * 3

        expr = x.sum()
        result = expr.forward()
        assert result.item() == pytest.approx(15.0)


# ============================================================================
# Type Validation Tests
# ============================================================================


class TestTypeValidation:
    """Test type validation in ProductExpression."""

    def test_can_multiply_variable_and_const(self, simple_variables):
        """Test that Variable * Const is allowed."""
        x, _ = simple_variables
        c = ConstExpression(5.0)

        # This should work
        z = x * c
        assert isinstance(z, ProductExpression)

    def test_can_multiply_sums_of_variables(self, simple_variables):
        """Test that (x+y) * (a+b) works if all are Variables."""
        x, y = simple_variables
        a = Variable(5, name="a")
        b = Variable(5, name="b")

        expr1 = x + y
        expr2 = a + b

        # This should work since tree only contains Variables and Consts
        z = expr1 * expr2
        assert isinstance(z, ProductExpression)

    def test_multiplication_works(self):
        """Test that multiplication of expressions works correctly."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")

        # These multiplications should work
        expr1 = x * y
        expr2 = (x + y) * (x - y)

        assert isinstance(expr1, ProductExpression)
        assert isinstance(expr2, ProductExpression)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_product_raises(self):
        """ProductExpression with no operands should raise."""
        with pytest.raises((ValueError, TypeError)):
            ProductExpression()

    def test_scalar_variable(self):
        """Test that scalar (0-d) variables work."""
        x = Variable(())  # Scalar
        x.value.data = torch.tensor(5.0)

        result = x.forward()
        assert result.item() == pytest.approx(5.0)

    def test_broadcasting_in_operations(self):
        """Test that broadcasting works correctly."""
        x = Variable(3, 1, name="x")
        y = Variable(1, 4, name="y")

        x.value.data = torch.ones(3, 1) * 2
        y.value.data = torch.ones(1, 4) * 3

        z = x * y  # Should broadcast to (3, 4)
        result = z.forward()

        assert result.shape == (3, 4)
        assert torch.allclose(result, torch.ones(3, 4) * 6.0)


# ============================================================================
# Module Integration Tests
# ============================================================================


class TestModuleIntegration:
    """Test that Expression works properly as a torch.nn.Module."""

    def test_can_register_as_submodule(self, simple_variables):
        """Test that expressions can be registered as submodules."""
        x, y = simple_variables
        expr = x + y

        # Create a parent module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expr = expr

        model = MyModule()
        params = list(model.parameters())
        assert len(params) == 2  # x and y

    def test_state_dict(self, simple_variables):
        """Test state_dict() includes all parameters with meaningful names."""
        x, y = simple_variables
        expr = x + y

        state = expr.state_dict()
        # Check we have the right number of parameters
        assert len(state) == 2

        # With Option 1, variable names should appear in keys
        param_keys = "".join(state.keys())
        assert x.name in param_keys
        assert y.name in param_keys

    def test_load_state_dict(self, simple_variables):
        """Test loading parameters from state dict."""
        x, y = simple_variables
        expr = x + y

        # Get actual keys from state dict
        state = expr.state_dict()
        keys = list(state.keys())

        # Find keys by variable name
        x_key = [k for k in keys if x.name in k][0]
        y_key = [k for k in keys if y.name in k][0]

        # Create new state
        new_state = {x_key: torch.ones(5) * 10, y_key: torch.ones(5) * 20}

        expr.load_state_dict(new_state, strict=False)

        # Verify the values were updated
        assert torch.equal(x.value, torch.ones(5) * 10)
        assert torch.equal(y.value, torch.ones(5) * 20)

    def test_to_device(self, simple_variables):
        """Test moving expression to different device."""
        x, y = simple_variables
        expr = x + y

        # Move to CPU explicitly
        expr = expr.to("cpu")
        result = expr.forward()
        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self, simple_variables):
        """Test moving expression to CUDA."""
        x, y = simple_variables
        expr = x + y

        expr = expr.to("cuda")
        result = expr.forward()
        assert result.device.type == "cuda"

    def test_train_eval_mode(self, simple_variables):
        """Test train/eval mode switching."""
        x, y = simple_variables
        expr = x + y

        # Should be in training mode by default
        assert expr.training

        expr.eval()
        assert not expr.training

        expr.train()
        assert expr.training


# ============================================================================
# Operator Split Tests (for AddExpression)
# ============================================================================


class TestOperatorSplit:
    """Test operator splitting for smooth/non-smooth decomposition."""

    def test_all_smooth_split(self, simple_variables):
        """Test splitting when all terms are smooth."""
        x, y = simple_variables
        z = Variable(5, name="z")

        expr = x + y + z  # All smooth
        smooth, non_smooth = expr.operator_split()

        assert smooth is not None
        assert smooth.is_smooth()
        assert non_smooth is None

    def test_mixed_split(self, simple_variables):
        """Test splitting with mixed smooth/non-smooth terms."""
        from rlaopt.atoms import L1Norm, SumSquares

        x, _ = simple_variables

        smooth_term = SumSquares(x)  # Smooth
        non_smooth_term = L1Norm(x)  # Non-smooth

        expr = smooth_term + non_smooth_term
        smooth, non_smooth = expr.operator_split()

        assert smooth is not None
        assert smooth.is_smooth()
        assert non_smooth is not None
        assert not non_smooth.is_smooth()

    def test_all_non_smooth_split(self):
        """Test splitting when all terms are non-smooth."""
        from rlaopt.atoms import L1Norm

        x = Variable(5, name="x")
        y = Variable(5, name="y")

        term1 = L1Norm(x)
        term2 = L1Norm(y)

        expr = term1 + term2
        smooth, non_smooth = expr.operator_split()

        assert smooth is None
        assert non_smooth is not None
        assert not non_smooth.is_smooth()


# ============================================================================
# Proximal Operator Tests
# ============================================================================


class TestProximalOperators:
    """Test proximal operator functionality."""

    def test_proxability_of_l1(self):
        """Test is_proxable for L1Norm."""
        from rlaopt.atoms import L1Norm

        x = Variable(5, name="x")

        # Single non-smooth term - proxable
        expr = L1Norm(x)
        assert expr.is_proxable()

    def test_disjoint_prox(self):
        """Test proxability with disjoint parameter sets."""
        from rlaopt.atoms import L1Norm

        x = Variable(5, name="x")
        y = Variable(5, name="y")

        # Disjoint non-smooth terms - proxable
        expr = L1Norm(x) + L1Norm(y)
        assert expr.is_proxable()

    def test_overlapping_prox_not_proxable(self):
        """Test that overlapping non-smooth terms are not proxable."""
        from rlaopt.atoms import L1Norm

        x = Variable(5, name="x")

        # Overlapping non-smooth terms - not proxable
        expr = L1Norm(x) + L1Norm(x)
        assert not expr.is_proxable()

    def test_prox_raises_if_not_proxable(self, simple_variables):
        """Test that calling prox on non-proxable expression raises."""
        x, y = simple_variables

        # Products are not proxable
        expr = x * y
        assert not expr.is_proxable()

        # Attempting to call prox should raise
        with pytest.raises(NotImplementedError):
            expr.prox(torch.ones(5), 1.0)


# ============================================================================
# Kwarg Filtering Tests
# ============================================================================


class TestKwargFiltering:
    """Test that NAryOperatorExpression filters kwargs correctly."""

    def test_kwarg_filtering_with_different_signatures(self):
        """Test that expressions receive only their relevant kwargs."""
        # This is a bit tricky to test directly since most expressions
        # don't take kwargs in forward(). This would be more relevant
        # for atoms that do take kwargs.

        x = Variable(5, name="x")
        y = Variable(5, name="y")

        # Both should work without kwargs
        expr = x + y
        result = expr.forward()
        assert result.shape == (5,)

    def test_forward_without_kwargs(self):
        """Test that forward works without any kwargs."""
        x = Variable(5, name="x")
        x.value.data = torch.ones(5) * 3

        result = x.forward()
        assert torch.equal(result, torch.ones(5) * 3)


# ============================================================================
# Special Methods Tests
# ============================================================================


class TestSpecialMethods:
    """Test special methods like __repr__ and __str__."""

    def test_variable_repr(self):
        """Test Variable.__repr__."""
        x = Variable(5, name="x")
        repr_str = repr(x)

        assert "Variable" in repr_str
        assert "x" in repr_str
        assert "shape" in repr_str.lower() or "(5,)" in repr_str

    def test_variable_str(self):
        """Test Variable.__str__."""
        x = Variable(5, name="x")
        str_repr = str(x)

        assert "Variable" in str_repr
        assert "x" in str_repr

    def test_expression_name(self, simple_variables):
        """Test _get_name() method."""
        x, y = simple_variables
        expr = x + y

        name = expr._get_name()
        assert "AddExpression" in name


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability of operations."""

    def test_large_values(self):
        """Test with large values."""
        x = Variable(5)
        x.value.data = torch.ones(5) * 1e6

        expr = x + x
        result = expr.forward()
        assert torch.allclose(result, torch.ones(5) * 2e6)

    def test_small_values(self):
        """Test with very small values."""
        x = Variable(5)
        x.value.data = torch.ones(5) * 1e-6

        expr = x * 2
        result = expr.forward()
        assert torch.allclose(result, torch.ones(5) * 2e-6)

    def test_mixed_scales(self):
        """Test with mixed scale values."""
        x = Variable(5)
        y = Variable(5)

        x.value.data = torch.ones(5) * 1e6
        y.value.data = torch.ones(5) * 1e-6

        expr = x + y
        result = expr.forward()
        # Should handle this gracefully
        assert torch.isfinite(result).all()


# ============================================================================
# Memory and Performance Tests
# ============================================================================


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_deep_expression_tree(self):
        """Test with deeply nested expression tree."""
        x = Variable(5)
        expr = x

        # Build deep tree: x + x + x + ... (100 times)
        for _ in range(100):
            expr = expr + x

        x.value.data = torch.ones(5)
        result = expr.forward()
        assert torch.equal(result, torch.ones(5) * 101)

    def test_wide_expression_tree(self):
        """Test with wide expression tree (many variables)."""
        variables = [Variable(5, name=f"x{i}") for i in range(50)]

        # Sum all variables
        expr = sum(variables[1:], variables[0])

        for v in variables:
            v.value.data = torch.ones(5)

        result = expr.forward()
        assert torch.equal(result, torch.ones(5) * 50)

    def test_parameter_count(self):
        """Test parameter counting in large expressions."""
        variables = [Variable(5, name=f"x{i}") for i in range(10)]
        expr = sum(variables[1:], variables[0])

        params = list(expr.parameters())
        assert len(params) == 10


# ============================================================================
# Expression Equivalence Tests
# ============================================================================


class TestExpressionEquivalence:
    """Test that mathematically equivalent expressions give same results."""

    def test_associativity_of_addition(self):
        """Test (x + y) + z == x + (y + z)."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")
        z = Variable(5, name="z")

        x.value.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y.value.data = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        z.value.data = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])

        expr1 = (x + y) + z
        expr2 = x + (y + z)

        result1 = expr1.forward()
        result2 = expr2.forward()

        assert torch.equal(result1, result2)

    def test_commutativity_of_addition(self):
        """Test x + y == y + x."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")

        x.value.data = torch.randn(5)
        y.value.data = torch.randn(5)

        expr1 = x + y
        expr2 = y + x

        result1 = expr1.forward()
        result2 = expr2.forward()

        assert torch.equal(result1, result2)

    def test_distributivity(self):
        """Test a * (x + y) == a*x + a*y."""
        x = Variable(5, name="x")
        y = Variable(5, name="y")

        x.value.data = torch.randn(5)
        y.value.data = torch.randn(5)

        expr1 = 3 * (x + y)
        expr2 = 3 * x + 3 * y

        result1 = expr1.forward()
        result2 = expr2.forward()

        assert torch.allclose(result1, result2)


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
