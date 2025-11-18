"""Tests for Linop atom."""


import pytest
import torch

import linops as lo
from rlaopt.atoms.linop import Linop
from rlaopt.expression import Variable


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    x = Variable((5,), name="x")
    x.value.data = torch.ones(5)
    return x


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    X = Variable((4, 3), name="X")
    X.value.data = torch.ones(4, 3)
    return X


@pytest.fixture
def simple_linop():
    """Create simple linop transformation data."""
    return lo.IdentityOperator(5)


@pytest.fixture
def nontrivial_linop():
    """Create non-trivial linop transformation data."""
    return lo.DiagonalOperator((1. + torch.arange(5))**-4)


class TestLinop:
    """Test Linop transformation atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_variable(self, vector_var, simple_linop):
        """Test initialization with Variable input."""
        linop = Linop(simple_linop, vector_var)
        assert linop.op is not None

    def test_init_with_expression_input(self, vector_var, simple_linop):
        """Test initialization with Expression input."""
        # Create an expression from a variable
        expr = vector_var + 1.0
        linop = Linop(simple_linop, expr)
        assert linop.op is not None

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_with_identity_transformation(self, vector_var, simple_linop):
        """Test forward() with identity transformation (A=I, b=0)."""
        linop = Linop(simple_linop, vector_var)
        result = linop.forward()
        expected = torch.ones(5)
        assert torch.allclose(result, expected)

    def test_forward_computes_linop_transformation(
        self, vector_var, nontrivial_linop
    ):
        """Test forward() correctly computes A @ x + b."""
        linop = Linop(nontrivial_linop, vector_var)
        result = linop.forward()

        expected = ((1. + torch.arange(5))**-1)**4
        assert torch.allclose(result, expected)

    def test_forward_with_expression_input(self, vector_var):
        """Test forward() with Expression input evaluates the expression first."""
        # Create expression: x + 2
        expr = vector_var + 2.0
        A = lo.IdentityOperator(5) * 2
        linop = Linop(A, expr)

        # vector_var.value = ones(5), so expr = ones(5) + 2 = 3*ones(5)
        # Result: 2*I @ 3*ones(5) = 6*ones(5)
        result = linop.forward()
        expected = torch.ones(5) * 6
        assert torch.allclose(result, expected)

    def test_forward_with_matrix_input(self, matrix_var):
        """Test forward() works with matrix inputs."""
        # Flatten transformation: sum rows
        A = lo.aslinearoperator(torch.ones(3, 4))  # Each row sums the 4 columns
        linop = Linop(A, matrix_var)

        # matrix_var is 4x3 of ones, A is 3x4, so A @ X gives 3x3
        # Each element = sum of row of A @ column of X = 4 * 1 = 4
        result = linop.forward()
        assert result.shape == (3, 3)

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_true(self, vector_var, simple_linop):
        """Test linop transformation is smooth."""
        linop = Linop(simple_linop, vector_var)
        assert linop.is_smooth() is True

    def test_is_proxable_returns_false(self, vector_var, simple_linop):
        """Test linop transformation is not proxable."""
        linop = Linop(simple_linop, vector_var)
        assert linop.is_proxable() is False

    def test_is_subsamplable_returns_true(self, vector_var, simple_linop):
        """Test linop transformation is subsamplable."""
        linop = Linop(simple_linop, vector_var)
        assert linop.is_subsamplable() is True

    def test_prox_raises_not_implemented(self, vector_var, simple_linop):
        """Test prox() raises NotImplementedError."""
        linop = Linop(simple_linop, vector_var)
        with pytest.raises(NotImplementedError, match="not proxable"):
            linop.prox(torch.ones(5), 1.0)

    def test_subsample(self, vector_var, simple_linop):
        """Test subsample() works."""
        linop = Linop(simple_linop, vector_var).subsample(torch.tensor([1, 3]))
        out = linop.forward()
        assert out.shape == (2,)
