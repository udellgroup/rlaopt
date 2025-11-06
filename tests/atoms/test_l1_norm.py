"""Tests for L1Norm atom."""

import pytest
import torch

from rlaopt.atoms.l1_norm import L1Norm
from rlaopt.expression import Variable


@pytest.fixture
def simple_variable():
    """Creates a simple 1D variable for testing."""
    return Variable(torch.tensor([1.0, -2.0, 3.0]))


@pytest.fixture
def matrix_variable():
    """Creates a 2D variable for testing."""
    return Variable(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))


class TestL1NormInitialization:
    """Tests for L1Norm initialization."""

    def test_init_with_float_scaling(self, simple_variable):
        """Test initialization with float scaling."""
        atom = L1Norm(simple_variable, scaling=2.0)
        assert torch.isclose(atom.scaling, torch.tensor(2.0))

    def test_init_with_tensor_scaling(self, simple_variable):
        """Test initialization with tensor scaling."""
        scaling = torch.tensor(3.5)
        atom = L1Norm(simple_variable, scaling=scaling)
        assert torch.isclose(atom.scaling, torch.tensor(3.5))

    def test_init_with_parameter_scaling(self, simple_variable):
        """Test initialization with nn.Parameter scaling."""
        scaling = torch.nn.Parameter(torch.tensor(1.5))
        atom = L1Norm(simple_variable, scaling=scaling)
        assert isinstance(atom.scaling, torch.nn.Parameter)
        assert torch.isclose(atom.scaling, torch.tensor(1.5))


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
        location = torch.tensor([2.0, -3.0, 0.5])
        result = atom.prox(location, prox_scaling=1.0)
        expected = torch.tensor([1.0, -2.0, 0.0])
        assert torch.allclose(result, expected)

    def test_prox_with_scaling_factor(self, simple_variable):
        """Test proximal operator with non-unit scaling."""
        atom = L1Norm(simple_variable, scaling=2.0)
        location = torch.tensor([5.0, -5.0, 1.0])
        result = atom.prox(location, prox_scaling=1.0)
        # threshold = 2.0 * 1.0 = 2.0
        expected = torch.tensor([3.0, -3.0, 0.0])
        assert torch.allclose(result, expected)

    def test_prox_all_below_threshold(self, simple_variable):
        """Test proximal operator when all values below threshold."""
        atom = L1Norm(simple_variable, scaling=1.0)
        location = torch.tensor([0.5, -0.3, 0.2])
        result = atom.prox(location, prox_scaling=1.0)
        expected = torch.zeros(3)
        assert torch.allclose(result, expected)


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

    def test_is_subsamplable_returns_false(self, simple_variable):
        """Test that L1-norm is not subsamplable."""
        atom = L1Norm(simple_variable, scaling=1.0)
        assert atom.is_subsamplable() is False

    def test_subsample_raises_error(self, simple_variable):
        """Test that subsample raises NotImplementedError."""
        atom = L1Norm(simple_variable, scaling=1.0)
        with pytest.raises(NotImplementedError, match="cannot be subsampled"):
            atom.subsample([0, 1])
