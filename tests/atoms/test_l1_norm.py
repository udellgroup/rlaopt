"""Tests for L1Norm atom."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import SumSquares
from rlaopt.atoms.l1_norm import L1Norm
from rlaopt.expression import ExprTree, Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def simple_variable():
    """Creates a simple 1D variable for testing."""
    return Variable(torch.tensor([1.0, -2.0, 3.0]), name="x")


@pytest.fixture
def matrix_variable():
    """Creates a 2D variable for testing."""
    return Variable(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))


class TestL1NormInitialization:
    """Tests for L1Norm initialization."""

    def test_init_with_float_scaling(self, simple_variable):
        """Test initialization with float scaling."""
        atom = L1Norm(simple_variable, scaling=2.0)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(2.0))

    def test_init_with_tensor_scaling(self, simple_variable):
        """Test initialization with tensor scaling."""
        scaling = torch.tensor(3.5)
        atom = L1Norm(simple_variable, scaling=scaling)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(3.5))

    def test_init_with_parameter_scaling(self, simple_variable):
        """Test initialization with nn.Parameter scaling."""
        scaling = torch.nn.Parameter(torch.tensor(1.5))
        atom = L1Norm(simple_variable, scaling=scaling)
        assert isinstance(atom.get_buffer("scaling"), torch.nn.Parameter)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(1.5))


class TestL1NormForward:
    """Tests for L1Norm forward evaluation."""

    def test_forward_unit_scaling(self, simple_variable):
        """Test forward pass with unit scaling."""
        atom = L1Norm(simple_variable, scaling=1.0)
        result = atom.forward()
        expected = torch.tensor(6.0)  # |1| + |-2| + |3| = 6
        assert torch.isclose(result, expected)

    def test_forward_with_scaling(self, simple_variable):
        """Test forward pass with non-unit scaling."""
        atom = L1Norm(simple_variable, scaling=2.0)
        result = atom.forward()
        expected = torch.tensor(12.0)  # 2 * (|1| + |-2| + |3|) = 12
        assert torch.isclose(result, expected)

    def test_forward_2d_tensor(self, matrix_variable):
        """Test forward pass with 2D tensor."""
        atom = L1Norm(matrix_variable, scaling=1.0)
        result = atom.forward()
        expected = torch.tensor(10.0)  # |1| + |-2| + |3| + |-4| = 10
        assert torch.isclose(result, expected)


class TestL1NormProx:
    """Tests for L1Norm proximal operator."""

    def test_prox_soft_threshold_basic(self, simple_variable):
        """Test proximal operator performs soft thresholding."""
        atom = L1Norm(simple_variable, scaling=1.0)
        location = TensorDict({simple_variable.name: torch.tensor([2.0, -3.0, 0.5])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.tensor([1.0, -2.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_with_scaling_factor(self, simple_variable):
        """Test proximal operator with non-unit scaling."""
        atom = L1Norm(simple_variable, scaling=2.0)
        location = TensorDict({simple_variable.name: torch.tensor([5.0, -5.0, 1.0])})
        result = atom.prox(location, prox_scaling=1.0)
        # threshold = 2.0 * 1.0 = 2.0
        expected = TensorDict({simple_variable.name: torch.tensor([3.0, -3.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_all_below_threshold(self, simple_variable):
        """Test proximal operator when all values below threshold."""
        atom = L1Norm(simple_variable, scaling=1.0)
        location = TensorDict({simple_variable.name: torch.tensor([0.5, -0.3, 0.2])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.zeros(3)})
        assert_allclose_td(result, expected)


class TestL1NormProperties:
    """Tests for L1Norm property methods."""

    def test_is_smooth_returns_false(self, simple_variable):
        """Test that L1-norm is not smooth."""
        atom = L1Norm(simple_variable, scaling=1.0)
        assert atom.is_smooth() is False

    def test_is_proxable_returns_true(self, simple_variable):
        """Test that L1-norm is proxable."""
        atom = L1Norm(simple_variable, scaling=1.0)
        assert atom.is_proxable() is True


class TestL1NormScaling:
    """Tests for L1Norm scalar multiplication."""

    def test_scalar_multiplication_optimization(self, simple_variable):
        """Test that scalar multiplication optimizes to scaled L1Norm."""
        atom = L1Norm(simple_variable, scaling=2.0)
        scaled = 3.0 * atom

        assert isinstance(scaled, L1Norm)
        assert scaled.tree() == ExprTree("L1Norm", ExprTree("Variable(x)"))
        assert torch.allclose(scaled.forward(), torch.tensor(36.0))


class TestL1NormDecompose:
    """Tests for L1Norm decompose method."""

    def test_decompose_with_variable_input(self, simple_variable):
        """Test decompose with a Variable input."""
        atom = L1Norm(simple_variable, scaling=2.0)
        decompositions = atom.decompose()

        # Should return a list with one decomposition
        assert len(decompositions) == 1
        decomp = decompositions[0]

        # Check that the decomposed atom is an L1Norm
        assert isinstance(decomp.atom, L1Norm)

        # Check that the new atom has the same scaling
        assert torch.allclose(decomp.atom.get_buffer("scaling"), torch.tensor(2.0))

        # The new atom should have a different variable
        new_var = decomp.atom.get_input("x")
        assert new_var is not simple_variable
        assert isinstance(new_var, Variable)

    def test_decompose_with_affine_expression(self):
        """Test decompose with an affine expression input."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([1.0, 2.0])
        affine_expr = A @ x + b

        atom = L1Norm(affine_expr, scaling=1.5)
        decompositions = atom.decompose()

        # Should return a list with one decomposition
        assert len(decompositions) == 1
        decomp = decompositions[0]

        # Check that the decomposed atom is an L1Norm
        assert isinstance(decomp.atom, L1Norm)

        # Check that the affine expression is preserved
        assert decomp.affine_expr is affine_expr

        # Check that the new atom has the same scaling
        assert torch.allclose(decomp.atom.get_buffer("scaling"), torch.tensor(1.5))

    def test_decompose_with_non_affine_expression(self):
        """Test decompose with a non-affine expression returns None."""
        x = Variable((3,), name="x")
        non_affine_expr = SumSquares(x)

        atom = L1Norm(non_affine_expr, scaling=1.0)
        decompositions = atom.decompose()

        # Should return None for non-affine input
        assert decompositions is None
