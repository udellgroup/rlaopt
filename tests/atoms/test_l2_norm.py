"""Tests for L2Norm atom."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import SumSquares
from rlaopt.atoms.l2_norm import L2Norm
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


class TestL2NormInitialization:
    """Tests for L2Norm initialization."""

    def test_init_with_float_scaling(self, simple_variable):
        """Test initialization with float scaling."""
        atom = L2Norm(simple_variable, scaling=2.0)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(2.0))

    def test_init_with_tensor_scaling(self, simple_variable):
        """Test initialization with tensor scaling."""
        scaling = torch.tensor(3.5)
        atom = L2Norm(simple_variable, scaling=scaling)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(3.5))

    def test_init_with_parameter_scaling(self, simple_variable):
        """Test initialization with nn.Parameter scaling."""
        scaling = torch.nn.Parameter(torch.tensor(1.5))
        atom = L2Norm(simple_variable, scaling=scaling)
        assert isinstance(atom.get_buffer("scaling"), torch.nn.Parameter)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(1.5))


class TestL2NormForward:
    """Tests for L2Norm forward evaluation."""

    def test_forward_unit_scaling(self, simple_variable):
        """Test forward pass with unit scaling."""
        atom = L2Norm(simple_variable, scaling=1.0)
        result = atom.forward()
        expected = torch.sqrt(torch.tensor(14.0))  # sqrt(1^2 + (-2)^2 + 3^2) = sqrt(14)
        assert torch.isclose(result, expected)

    def test_forward_with_scaling(self, simple_variable):
        """Test forward pass with non-unit scaling."""
        atom = L2Norm(simple_variable, scaling=2.0)
        result = atom.forward()
        expected = 2.0 * torch.sqrt(torch.tensor(14.0))
        assert torch.isclose(result, expected)

    def test_forward_2d_tensor(self, matrix_variable):
        """Test forward pass with 2D tensor."""
        atom = L2Norm(matrix_variable, scaling=1.0)
        result = atom.forward()
        # sqrt(1^2 + (-2)^2 + 3^2 + (-4)^2) = sqrt(30)
        expected = torch.sqrt(torch.tensor(30.0))
        assert torch.isclose(result, expected)


class TestL2NormProx:
    """Tests for L2Norm proximal operator."""

    def test_prox_block_soft_threshold_basic(self, simple_variable):
        """Test proximal operator performs block (L2) soft thresholding."""
        atom = L2Norm(simple_variable, scaling=1.0)
        # v has norm 5 -> factor = 1 - 1/5 = 0.8
        location = TensorDict({simple_variable.name: torch.tensor([3.0, 4.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict(
            {simple_variable.name: torch.tensor([2.4, 3.2, 0.0])}
        )
        assert_allclose_td(result, expected)

    def test_prox_with_scaling_factor(self, simple_variable):
        """Test proximal operator with non-unit scaling."""
        atom = L2Norm(simple_variable, scaling=2.0)
        # lam = 2.0 * 0.5 = 1.0, norm = 5 -> factor = 1 - 1/5 = 0.8
        location = TensorDict({simple_variable.name: torch.tensor([3.0, 4.0, 0.0])})
        result = atom.prox(location, prox_scaling=0.5)
        expected = TensorDict(
            {simple_variable.name: torch.tensor([2.4, 3.2, 0.0])}
        )
        assert_allclose_td(result, expected)

    def test_prox_all_below_threshold(self, simple_variable):
        """Test proximal operator when norm is below threshold."""
        atom = L2Norm(simple_variable, scaling=2.0)
        # v norm = sqrt(2) ~ 1.41, lam = 2 * 1 = 2.0 -> norm <= lam -> 0
        location = TensorDict({simple_variable.name: torch.tensor([1.0, 1.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.zeros(3)})
        assert_allclose_td(result, expected)


class TestL2NormProperties:
    """Tests for L2Norm property methods."""

    def test_is_smooth_returns_false(self, simple_variable):
        """Test that L2-norm is not smooth."""
        atom = L2Norm(simple_variable, scaling=1.0)
        assert atom.is_smooth() is False

    def test_is_proxable_returns_true(self, simple_variable):
        """Test that L2-norm is proxable."""
        atom = L2Norm(simple_variable, scaling=1.0)
        assert atom.is_proxable() is True


class TestL2NormScaling:
    """Tests for L2Norm scalar multiplication."""

    def test_scalar_multiplication_optimization(self, simple_variable):
        """Test that scalar multiplication optimizes to scaled L2Norm."""
        atom = L2Norm(simple_variable, scaling=2.0)
        scaled = 3.0 * atom

        assert isinstance(scaled, L2Norm)
        assert scaled.tree() == ExprTree("L2Norm", ExprTree("Variable(x)"))

        # Original scaling 2.0, multiplied by 3.0 -> effective scaling 6.0
        expected = 6.0 * torch.sqrt(torch.tensor(14.0))
        assert torch.allclose(scaled.forward(), expected)


class TestL2NormDecompose:
    """Tests for L2Norm decompose method."""

    def test_decompose_with_variable_input(self, simple_variable):
        """Test decompose with a Variable input."""
        atom = L2Norm(simple_variable, scaling=2.0)
        decompositions = atom.decompose()

        # Should return a list with one decomposition
        assert len(decompositions) == 1
        decomp = decompositions[0]

        # Check the decomposed atom is an L2Norm
        assert isinstance(decomp.atom, L2Norm)

        # Check the new atom has the same scaling
        assert torch.allclose(decomp.atom.get_buffer("scaling"), torch.tensor(2.0))

        # New atom should have a different variable
        new_var = decomp.atom.get_input("x")
        assert new_var is not simple_variable
        assert isinstance(new_var, Variable)

    def test_decompose_with_affine_expression(self):
        """Test decompose with an affine expression input."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([1.0, 2.0])
        affine_expr = A @ x + b

        atom = L2Norm(affine_expr, scaling=1.5)
        decompositions = atom.decompose()

        # Should return a list with one decomposition
        assert len(decompositions) == 1
        decomp = decompositions[0]

        assert isinstance(decomp.atom, L2Norm)
        
        assert decomp.affine_expr is affine_expr

        assert torch.allclose(decomp.atom.get_buffer("scaling"), torch.tensor(1.5))

    def test_decompose_with_non_affine_expression(self):
        """Test decompose with a non-affine expression returns None."""
        x = Variable((3,), name="x")
        non_affine_expr = SumSquares(x)

        atom = L2Norm(non_affine_expr, scaling=1.0)
        decompositions = atom.decompose()

        # Should return None for non-affine input
        assert decompositions is None