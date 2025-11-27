"""Tests for ElasticNet atom."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import SumSquares
from rlaopt.atoms.elastic_net import ElasticNet
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


class TestElasticNetInitialization:
    """Tests for ElasticNet initialization."""

    def test_init_default_scaling(self, simple_variable):
        """Test initialization with default scaling factors."""
        atom = ElasticNet(simple_variable)
        assert torch.isclose(atom.get_buffer("l1_scaling"), torch.tensor(1.0))
        assert torch.isclose(atom.get_buffer("l2_scaling"), torch.tensor(1.0))

    def test_init_custom_scaling(self, simple_variable):
        """Test initialization with custom scaling factors."""
        atom = ElasticNet(simple_variable, l1_scaling=0.5, l2_scaling=2.0)
        assert torch.isclose(atom.get_buffer("l1_scaling"), torch.tensor(0.5))
        assert torch.isclose(atom.get_buffer("l2_scaling"), torch.tensor(2.0))


class TestElasticNetForward:
    """Tests for ElasticNet forward evaluation."""

    def test_forward_equal_weights(self, simple_variable):
        """Test forward pass with equal L1 and L2 weights."""
        # x = [1, -2, 3]: L1 = 6, L2^2 = 14
        atom = ElasticNet(simple_variable, l1_scaling=1.0, l2_scaling=1.0)
        result = atom.forward()
        expected = 1.0 * 6.0 + (1.0 / 2) * 14.0  # 6 + 7 = 13
        assert torch.isclose(result, torch.tensor(expected))

    def test_forward_lasso_like(self, simple_variable):
        """Test forward pass with L1 emphasis (lasso-like)."""
        atom = ElasticNet(simple_variable, l1_scaling=1.0, l2_scaling=0.1)
        result = atom.forward()
        expected = 1.0 * 6.0 + (0.1 / 2) * 14.0  # 6 + 0.7 = 6.7
        assert torch.isclose(result, torch.tensor(expected))

    def test_forward_ridge_like(self, simple_variable):
        """Test forward pass with L2 emphasis (ridge-like)."""
        atom = ElasticNet(simple_variable, l1_scaling=0.1, l2_scaling=1.0)
        result = atom.forward()
        expected = 0.1 * 6.0 + (1.0 / 2) * 14.0  # 0.6 + 7 = 7.6
        assert torch.isclose(result, torch.tensor(expected))

    def test_forward_2d_tensor(self, matrix_variable):
        """Test forward pass with 2D tensor."""
        # [[1, -2], [3, -4]]: L1 = 10, L2^2 = 30
        atom = ElasticNet(matrix_variable, l1_scaling=1.0, l2_scaling=1.0)
        result = atom.forward()
        expected = 1.0 * 10.0 + (1.0 / 2) * 30.0  # 10 + 15 = 25
        assert torch.isclose(result, torch.tensor(expected))


class TestElasticNetProx:
    """Tests for ElasticNet proximal operator."""

    def test_prox_basic(self, simple_variable):
        """Test proximal operator with unit scaling."""
        atom = ElasticNet(simple_variable, l1_scaling=1.0, l2_scaling=1.0)
        location = TensorDict({simple_variable.name: torch.tensor([3.0, -4.0, 1.0])})
        result = atom.prox(location, prox_scaling=1.0)
        # threshold = 1.0, l2_term = 1 + 1*1 = 2
        # soft_threshold([3, -4, 1], 1) = [2, -3, 0]
        # result = [2, -3, 0] / 2 = [1, -1.5, 0]
        expected = TensorDict({simple_variable.name: torch.tensor([1.0, -1.5, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_lasso_like(self, simple_variable):
        """Test proximal operator with L1 emphasis."""
        atom = ElasticNet(simple_variable, l1_scaling=1.0, l2_scaling=0.0)
        location = TensorDict({simple_variable.name: torch.tensor([2.0, -3.0, 0.5])})
        result = atom.prox(location, prox_scaling=1.0)
        # Pure soft-thresholding when l2_scaling=0
        expected = TensorDict({simple_variable.name: torch.tensor([1.0, -2.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_ridge_like(self, simple_variable):
        """Test proximal operator with L2 emphasis."""
        atom = ElasticNet(simple_variable, l1_scaling=0.0, l2_scaling=2.0)
        location = TensorDict({simple_variable.name: torch.tensor([4.0, -6.0, 2.0])})
        result = atom.prox(location, prox_scaling=1.0)
        # No thresholding when l1_scaling=0, just scaling by 1/(1+2*1)=1/3
        expected = TensorDict(
            {simple_variable.name: torch.tensor([4.0 / 3, -2.0, 2.0 / 3])}
        )
        assert_allclose_td(result, expected)


class TestElasticNetAbstractMethods:
    """Tests for ElasticNet abstract method implementations."""

    def test_is_smooth_returns_false(self, simple_variable):
        """Test that elastic net is not smooth due to L1 component."""
        atom = ElasticNet(simple_variable)
        assert atom.is_smooth() is False

    def test_is_proxable_returns_true(self, simple_variable):
        """Test that elastic net is proxable."""
        atom = ElasticNet(simple_variable)
        assert atom.is_proxable() is True


class TestElasticNetScaling:
    """Tests for ElasticNet scalar multiplication."""

    def test_scalar_multiplication_optimization(self, simple_variable):
        """Test that scalar multiplication optimizes to scaled ElasticNet."""
        atom = ElasticNet(simple_variable, l1_scaling=1.0, l2_scaling=1.0)
        scaled = 2.0 * atom

        assert isinstance(scaled, ElasticNet)
        assert scaled.tree() == ExprTree("ElasticNet", ExprTree("Variable(x)"))
        assert torch.allclose(scaled.forward(), torch.tensor(26.0))


class TestElasticNetDecompose:
    """Tests for ElasticNet decompose method."""

    def test_decompose_with_variable_input(self, simple_variable):
        """Test decompose with a Variable input."""
        atom = ElasticNet(simple_variable, l1_scaling=0.5, l2_scaling=2.0)
        decompositions = atom.decompose()

        # Should return a list with one decomposition
        assert len(decompositions) == 1
        decomp = decompositions[0]

        # Check that the decomposed atom is an ElasticNet
        assert isinstance(decomp.atom, ElasticNet)

        # Check that the new atom has the same scaling parameters
        assert torch.allclose(decomp.atom.get_buffer("l1_scaling"), torch.tensor(0.5))
        assert torch.allclose(decomp.atom.get_buffer("l2_scaling"), torch.tensor(2.0))

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

        atom = ElasticNet(affine_expr, l1_scaling=1.0, l2_scaling=0.5)
        decompositions = atom.decompose()

        # Should return a list with one decomposition
        assert len(decompositions) == 1
        decomp = decompositions[0]

        # Check that the decomposed atom is an ElasticNet
        assert isinstance(decomp.atom, ElasticNet)

        # Check that the affine expression is preserved
        assert decomp.affine_expr is affine_expr

        # Check that the new atom has the same scaling parameters
        assert torch.allclose(decomp.atom.get_buffer("l1_scaling"), torch.tensor(1.0))
        assert torch.allclose(decomp.atom.get_buffer("l2_scaling"), torch.tensor(0.5))

    def test_decompose_with_non_affine_expression(self):
        """Test decompose with a non-affine expression returns None."""
        x = Variable((3,), name="x")
        non_affine_expr = SumSquares(x)

        atom = ElasticNet(non_affine_expr, l1_scaling=1.0, l2_scaling=1.0)
        decompositions = atom.decompose()

        # Should return None for non-affine input
        assert decompositions is None
