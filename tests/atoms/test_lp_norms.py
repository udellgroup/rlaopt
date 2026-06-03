"""Tests for the Lp-norm regularizer atoms."""

from abc import ABC, abstractmethod

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import L1Norm, L2Norm, LInfNorm, SumSquares
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


class BaseLpNormTest(ABC):
    """Shared test suite for the Lp-norm regularizer atoms.

    Concrete subclasses bind the atom class under test (``atom_cls``) and the
    unit-scaling norm of the shared fixtures (``expected_norm_1d`` /
    ``expected_norm_2d``), and add their own norm-specific proximal-operator
    tests. The base class itself is not collected by pytest (it is not named
    ``Test*``).
    """

    @property
    @abstractmethod
    def atom_cls(self) -> type:
        """The Lp-norm atom class under test."""
        ...

    @property
    @abstractmethod
    def expected_norm_1d(self) -> float:
        """Unit-scaling norm of the ``simple_variable`` value [1, -2, 3]."""
        ...

    @property
    @abstractmethod
    def expected_norm_2d(self) -> float:
        """Unit-scaling norm of the ``matrix_variable`` value [[1,-2],[3,-4]]."""
        ...

    def test_init_with_float_scaling(self, simple_variable):
        """Test initialization with float scaling."""
        atom = self.atom_cls(simple_variable, scaling=2.0)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(2.0))

    def test_init_with_tensor_scaling(self, simple_variable):
        """Test initialization with tensor scaling."""
        atom = self.atom_cls(simple_variable, scaling=torch.tensor(3.5))
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(3.5))

    def test_init_with_parameter_scaling(self, simple_variable):
        """Test initialization with nn.Parameter scaling."""
        scaling = torch.nn.Parameter(torch.tensor(1.5))
        atom = self.atom_cls(simple_variable, scaling=scaling)
        assert isinstance(atom.get_buffer("scaling"), torch.nn.Parameter)
        assert torch.isclose(atom.get_buffer("scaling"), torch.tensor(1.5))

    def test_forward_unit_scaling(self, simple_variable):
        """Test forward pass with unit scaling."""
        atom = self.atom_cls(simple_variable, scaling=1.0)
        assert torch.isclose(atom.forward(), torch.tensor(self.expected_norm_1d))

    def test_forward_with_scaling(self, simple_variable):
        """Test forward pass with non-unit scaling."""
        atom = self.atom_cls(simple_variable, scaling=2.0)
        expected = torch.tensor(2.0 * self.expected_norm_1d)
        assert torch.isclose(atom.forward(), expected)

    def test_forward_2d_tensor(self, matrix_variable):
        """Test forward pass with a 2D variable."""
        atom = self.atom_cls(matrix_variable, scaling=1.0)
        assert torch.isclose(atom.forward(), torch.tensor(self.expected_norm_2d))

    def test_is_smooth_returns_false(self, simple_variable):
        """Test that the norm is not smooth."""
        assert self.atom_cls(simple_variable).is_smooth() is False

    def test_is_proxable_returns_true(self, simple_variable):
        """Test that the norm is proxable for a Variable input."""
        assert self.atom_cls(simple_variable).is_proxable() is True

    def test_scalar_multiplication_optimization(self, simple_variable):
        """Test that scalar multiplication folds into a scaled norm atom."""
        atom = self.atom_cls(simple_variable, scaling=2.0)
        scaled = 3.0 * atom
        assert isinstance(scaled, self.atom_cls)
        assert scaled.tree() == ExprTree(
            self.atom_cls.__name__, ExprTree("Variable(x)")
        )
        # Effective scaling 2.0 * 3.0 = 6.0.
        assert torch.allclose(
            scaled.forward(), torch.tensor(6.0 * self.expected_norm_1d)
        )

    def test_decompose_with_variable_input(self, simple_variable):
        """Test decompose with a Variable input."""
        atom = self.atom_cls(simple_variable, scaling=2.0)
        decompositions = atom.decompose()
        assert len(decompositions) == 1
        decomp = decompositions[0]
        assert isinstance(decomp.atom, self.atom_cls)
        assert torch.allclose(decomp.atom.get_buffer("scaling"), torch.tensor(2.0))
        new_var = decomp.atom.get_input("x")
        assert new_var is not simple_variable
        assert isinstance(new_var, Variable)

    def test_decompose_with_affine_expression(self):
        """Test decompose with an affine expression input."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([1.0, 2.0])
        affine_expr = A @ x + b
        atom = self.atom_cls(affine_expr, scaling=1.5)
        decompositions = atom.decompose()
        assert len(decompositions) == 1
        decomp = decompositions[0]
        assert isinstance(decomp.atom, self.atom_cls)
        assert decomp.affine_expr is affine_expr
        assert torch.allclose(decomp.atom.get_buffer("scaling"), torch.tensor(1.5))

    def test_decompose_with_non_affine_expression(self):
        """Test decompose with a non-affine expression returns None."""
        x = Variable((3,), name="x")
        atom = self.atom_cls(SumSquares(x), scaling=1.0)
        assert atom.decompose() is None

    def test_prox_differentiable_wrt_scaling(self, simple_variable):
        """Test prox correctly differentiates w.r.t. the scaling."""
        v = torch.tensor([3.0, -2.0, 0.5])

        def loss(scaling):
            atom = self.atom_cls(simple_variable, scaling=scaling)
            location = TensorDict({simple_variable.name: v})
            result = atom.prox(location, prox_scaling=1.0)
            return (result[simple_variable.name] ** 2).sum()

        scaling = torch.tensor(0.3)
        grad = torch.func.grad(loss)(scaling)
        numerical = (loss(scaling + 1e-3) - loss(scaling - 1e-3)) / 2e-3
        assert torch.isfinite(grad)
        assert grad != 0.0
        assert torch.allclose(grad, numerical, rtol=0.05)


class TestL1Norm(BaseLpNormTest):
    """Tests for the L1Norm atom."""

    atom_cls = L1Norm
    expected_norm_1d = 6.0  # |1| + |-2| + |3|
    expected_norm_2d = 10.0  # |1| + |-2| + |3| + |-4|

    def test_prox_soft_threshold_basic(self, simple_variable):
        """Prox performs soft thresholding."""
        atom = L1Norm(simple_variable, scaling=1.0)
        location = TensorDict({simple_variable.name: torch.tensor([2.0, -3.0, 0.5])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.tensor([1.0, -2.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_with_scaling_factor(self, simple_variable):
        """Prox soft-thresholds at scaling * prox_scaling."""
        atom = L1Norm(simple_variable, scaling=2.0)
        location = TensorDict({simple_variable.name: torch.tensor([5.0, -5.0, 1.0])})
        result = atom.prox(location, prox_scaling=1.0)
        # threshold = 2.0 * 1.0 = 2.0
        expected = TensorDict({simple_variable.name: torch.tensor([3.0, -3.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_all_below_threshold(self, simple_variable):
        """Prox zeros out everything below the threshold."""
        atom = L1Norm(simple_variable, scaling=1.0)
        location = TensorDict({simple_variable.name: torch.tensor([0.5, -0.3, 0.2])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.zeros(3)})
        assert_allclose_td(result, expected)


class TestL2Norm(BaseLpNormTest):
    """Tests for the L2Norm atom."""

    atom_cls = L2Norm
    expected_norm_1d = 14.0**0.5  # sqrt(1 + 4 + 9)
    expected_norm_2d = 30.0**0.5  # sqrt(1 + 4 + 9 + 16)

    def test_prox_block_soft_threshold_basic(self, simple_variable):
        """Prox performs block (L2) soft thresholding."""
        atom = L2Norm(simple_variable, scaling=1.0)
        # v has norm 5 -> factor = 1 - 1/5 = 0.8
        location = TensorDict({simple_variable.name: torch.tensor([3.0, 4.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.tensor([2.4, 3.2, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_with_scaling_factor(self, simple_variable):
        """Prox uses lam = scaling * prox_scaling."""
        atom = L2Norm(simple_variable, scaling=2.0)
        # lam = 2.0 * 0.5 = 1.0, norm = 5 -> factor = 1 - 1/5 = 0.8
        location = TensorDict({simple_variable.name: torch.tensor([3.0, 4.0, 0.0])})
        result = atom.prox(location, prox_scaling=0.5)
        expected = TensorDict({simple_variable.name: torch.tensor([2.4, 3.2, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_all_below_threshold(self, simple_variable):
        """Prox returns 0 when the norm is below the threshold."""
        atom = L2Norm(simple_variable, scaling=2.0)
        # v norm = sqrt(2) ~ 1.41, lam = 2 * 1 = 2.0 -> norm <= lam -> 0
        location = TensorDict({simple_variable.name: torch.tensor([1.0, 1.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.zeros(3)})
        assert_allclose_td(result, expected)


class TestLInfNorm(BaseLpNormTest):
    """Tests for the LInfNorm atom."""

    atom_cls = LInfNorm
    expected_norm_1d = 3.0  # max(|1|, |-2|, |3|)
    expected_norm_2d = 4.0  # max(|1|, |-2|, |3|, |-4|)

    def test_prox_basic(self, simple_variable):
        """Prox is v minus the projection onto the L1 ball (dual of Linf)."""
        atom = LInfNorm(simple_variable, scaling=1.0)
        # lam = 1.0; proj_{||.||_1 <= 1}([3, -1, 0]) = [1, 0, 0]
        location = TensorDict({simple_variable.name: torch.tensor([3.0, -1.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.tensor([2.0, -1.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_with_scaling_factor(self, simple_variable):
        """Prox uses lam = scaling * prox_scaling for the dual projection."""
        atom = LInfNorm(simple_variable, scaling=2.0)
        # lam = 2.0; proj_{||.||_1 <= 2}([3, -1, 0]) = [2, 0, 0]
        location = TensorDict({simple_variable.name: torch.tensor([3.0, -1.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.tensor([1.0, -1.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_zero_when_l1_below_lambda(self, simple_variable):
        """Prox returns 0 when ||v||_1 <= lam (the L-infinity prox null region)."""
        atom = LInfNorm(simple_variable, scaling=1.0)
        # ||[0.3, -0.2, 0.1]||_1 = 0.6 <= lam = 1.0 -> prox = 0
        location = TensorDict({simple_variable.name: torch.tensor([0.3, -0.2, 0.1])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({simple_variable.name: torch.zeros(3)})
        assert_allclose_td(result, expected)
