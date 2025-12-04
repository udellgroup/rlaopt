"""Tests for unary expression operations."""

import pytest
import torch

from rlaopt.expression import ExprTree
from rlaopt.expression.unary_expressions import Power, ReduceSum, Transpose
from rlaopt.expression.variable import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="x")


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    return Variable((3, 4), name="A")


class TestReduceSum:
    """Test ReduceSum unary operation."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_expression(self, vector_var):
        """Test initialization with an expression."""
        expr = ReduceSum(vector_var)
        assert expr.operand is vector_var
        assert expr.dim is None

    def test_init_with_dim(self, matrix_var):
        """Test initialization with dimension parameter."""
        expr = ReduceSum(matrix_var, dim=0)
        assert expr.dim == 0

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_sum_all_elements(self, vector_var):
        """Test forward() sums all elements when dim=None."""
        vector_var.value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = ReduceSum(vector_var)
        result = expr.forward()
        assert result.item() == 15.0

    def test_forward_sum_along_dimension(self, matrix_var):
        """Test forward() sums along specified dimension."""
        matrix_var.value = torch.ones(3, 4)
        expr = ReduceSum(matrix_var, dim=0)
        result = expr.forward()
        assert result.shape == torch.Size([4])
        assert torch.allclose(result, torch.tensor([3.0, 3.0, 3.0, 3.0]))

    def test_forward_sum_along_last_dimension(self, matrix_var):
        """Test forward() sums along last dimension."""
        matrix_var.value = torch.ones(3, 4)
        expr = ReduceSum(matrix_var, dim=1)
        result = expr.forward()
        assert result.shape == torch.Size([3])
        assert torch.allclose(result, torch.tensor([4.0, 4.0, 4.0]))

    # ----------------------
    # Smoothness and affineness tests
    # ----------------------

    def test_is_smooth(self, vector_var):
        """Test that ReduceSum is smooth when operand is smooth."""
        expr = ReduceSum(vector_var)
        assert expr.is_smooth() is True

    def test_is_affine_when_operand_is_affine(self, vector_var):
        """Test that ReduceSum is affine when operand is affine."""
        expr = ReduceSum(vector_var)
        assert expr.is_affine() is True

    # ----------------------
    # Tree representation tests
    # ----------------------

    def test_tree_without_dim(self, vector_var):
        """Test tree() representation without dimension."""
        expr = ReduceSum(vector_var)
        tree = expr.tree()
        assert tree == ExprTree("sum", vector_var.tree())

    def test_tree_with_dim(self, matrix_var):
        """Test tree() representation with dimension."""
        expr = ReduceSum(matrix_var, dim=0)
        tree = expr.tree()
        assert tree == ExprTree("sum_0", matrix_var.tree())


class TestTranspose:
    """Test Transpose unary operation."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_expression(self, matrix_var):
        """Test initialization with an expression."""
        expr = Transpose(matrix_var)
        assert expr.operand is matrix_var

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_transposes_matrix(self, matrix_var):
        """Test forward() transposes a matrix."""
        matrix_var.value = torch.arange(12).reshape(3, 4).float()
        expr = Transpose(matrix_var)
        result = expr.forward()
        assert torch.equal(result, matrix_var.value.transpose(-2, -1))

    def test_forward_transposes_last_two_dims(self):
        """Test forward() transposes last two dimensions for higher-dim tensors."""
        tensor_var = Variable(torch.randn(2, 3, 4))
        expr = Transpose(tensor_var)
        result = expr.forward()
        assert torch.equal(result, tensor_var.value.transpose(-2, -1))

    # ----------------------
    # Smoothness and affineness tests
    # ----------------------

    def test_is_smooth(self, matrix_var):
        """Test that Transpose is smooth when operand is smooth."""
        expr = Transpose(matrix_var)
        assert expr.is_smooth() is True

    def test_is_affine_when_operand_is_affine(self, matrix_var):
        """Test that Transpose is affine when operand is affine."""
        expr = Transpose(matrix_var)
        assert expr.is_affine() is True

    # ----------------------
    # Tree representation tests
    # ----------------------

    def test_tree(self, matrix_var):
        """Test tree() representation."""
        expr = Transpose(matrix_var)
        tree = expr.tree()
        assert tree == ExprTree("transpose", matrix_var.tree())


class TestPower:
    """Test Power unary operation."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_expression_and_exponent(self, vector_var):
        """Test initialization with expression and exponent."""
        expr = Power(vector_var, 2)
        assert expr.operand is vector_var
        assert expr.exponent == 2

    def test_init_with_float_exponent(self, vector_var):
        """Test initialization with float exponent."""
        expr = Power(vector_var, 2.5)
        assert expr.exponent == 2.5

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_with_exponent_two(self, vector_var):
        """Test forward() with exponent=2 squares elements."""
        vector_var.value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = Power(vector_var, 2)
        result = expr.forward()
        expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
        assert torch.allclose(result, expected)

    def test_forward_with_fractional_exponent(self, vector_var):
        """Test forward() with fractional exponent."""
        vector_var.value = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
        expr = Power(vector_var, 0.5)
        result = expr.forward()
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.allclose(result, expected)

    def test_forward_with_exponent_one(self, vector_var):
        """Test forward() with exponent=1 returns identity."""
        vector_var.value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = Power(vector_var, 1)
        result = expr.forward()
        assert torch.equal(result, vector_var.value)

    def test_forward_with_exponent_zero(self, vector_var):
        """Test forward() with exponent=0 returns all ones."""
        vector_var.value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = Power(vector_var, 0)
        result = expr.forward()
        expected = torch.ones(5)
        assert torch.allclose(result, expected)

    # ----------------------
    # Smoothness and affineness tests
    # ----------------------

    def test_is_smooth(self, vector_var):
        """Test that Power is smooth when operand is smooth."""
        expr = Power(vector_var, 2)
        assert expr.is_smooth() is True

    def test_is_affine_only_when_exponent_is_one(self, vector_var):
        """Test that Power is affine only when exponent=1."""
        expr_affine = Power(vector_var, 1)
        expr_not_affine = Power(vector_var, 2)
        assert expr_affine.is_affine() is True
        assert expr_not_affine.is_affine() is False

    # ----------------------
    # Tree representation tests
    # ----------------------

    def test_tree(self, vector_var):
        """Test tree() representation."""
        expr = Power(vector_var, 2)
        tree = expr.tree()
        assert tree == ExprTree("power_2", vector_var.tree())

    def test_tree_with_float_exponent(self, vector_var):
        """Test tree() representation with float exponent."""
        expr = Power(vector_var, 2.5)
        tree = expr.tree()
        assert tree == ExprTree("power_2.5", vector_var.tree())
