"""Tests for norm ball atoms."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import L1NormBall, L2NormBall, LInfNormBall, SumSquares
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((3,), name="x")


class TestL1NormBall:
    """Tests for L1NormBall atom."""

    def test_forward_satisfied_and_violated(self, vector_var):
        """Forward returns 0 if within radius and inf if violated."""
        atom = L1NormBall(vector_var, radius=2.0)
        vector_var.value = torch.tensor([1.0, -0.5, 0.0])  # L1 = 1.5
        assert torch.allclose(atom.forward(), torch.tensor(0.0))

        vector_var.value = torch.tensor([2.0, 1.0, 0.0])  # L1 = 3.0
        assert torch.isinf(atom.forward())

    def test_is_smooth_and_proxable(self, vector_var):
        """L1NormBall is nonsmooth and proxable for Variable input."""
        atom = L1NormBall(vector_var, radius=1.0)
        assert atom.is_smooth() is False
        assert atom.is_proxable() is True

    def test_is_proxable_false_for_expression(self, vector_var):
        """L1NormBall is not proxable when input is not a Variable."""
        A = torch.eye(3)
        b = torch.ones(3)
        atom = L1NormBall(A @ vector_var + b, radius=1.0)
        assert atom.is_proxable() is False

    def test_prox_projects_onto_ball(self, vector_var):
        """Prox projects infeasible points onto the L1 ball."""
        atom = L1NormBall(vector_var, radius=2.0)
        location = TensorDict({vector_var.name: torch.tensor([3.0, -1.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.tensor([2.0, -0.0, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_preserves_feasible_point(self, vector_var):
        """Prox returns input unchanged for feasible points."""
        atom = L1NormBall(vector_var, radius=2.0)
        location = TensorDict({vector_var.name: torch.tensor([0.5, -0.5, 0.5])})
        result = atom.prox(location, prox_scaling=1.0)
        assert_allclose_td(result, location)

    def test_decompose_affine_and_non_affine(self, vector_var):
        """Decompose works for affine inputs and returns None for non-affine."""
        A = torch.eye(3)
        b = torch.ones(3)
        affine_expr = A @ vector_var + b
        atom = L1NormBall(affine_expr, radius=1.5)
        decompositions = atom.decompose()
        assert decompositions is not None
        assert len(decompositions) == 1
        assert decompositions[0].affine_expr is affine_expr

        non_affine = SumSquares(vector_var)
        atom_non_affine = L1NormBall(non_affine, radius=1.0)
        assert atom_non_affine.decompose() is None


class TestL2NormBall:
    """Tests for L2NormBall atom."""

    def test_forward_satisfied_and_violated(self, vector_var):
        """Forward returns 0 if within radius and inf if violated."""
        atom = L2NormBall(vector_var, radius=2.0)
        vector_var.value = torch.tensor([1.0, 1.0, 0.0])  # L2 ~ 1.41
        assert torch.allclose(atom.forward(), torch.tensor(0.0))

        vector_var.value = torch.tensor([2.0, 2.0, 0.0])  # L2 ~ 2.83
        assert torch.isinf(atom.forward())

    def test_is_smooth_and_proxable(self, vector_var):
        """L2NormBall is nonsmooth and proxable for Variable input."""
        atom = L2NormBall(vector_var, radius=1.0)
        assert atom.is_smooth() is False
        assert atom.is_proxable() is True

    def test_prox_projects_onto_ball(self, vector_var):
        """Prox projects infeasible points onto the L2 ball."""
        atom = L2NormBall(vector_var, radius=2.0)
        location = TensorDict({vector_var.name: torch.tensor([3.0, 4.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.tensor([1.2, 1.6, 0.0])})
        assert_allclose_td(result, expected)

    def test_prox_preserves_feasible_point(self, vector_var):
        """Prox returns input unchanged for feasible points."""
        atom = L2NormBall(vector_var, radius=2.0)
        location = TensorDict({vector_var.name: torch.tensor([0.5, -0.5, 0.5])})
        result = atom.prox(location, prox_scaling=1.0)
        assert_allclose_td(result, location)


class TestLInfNormBall:
    """Tests for LInfNormBall atom."""

    def test_forward_satisfied_and_violated(self, vector_var):
        """Forward returns 0 if within radius and inf if violated."""
        atom = LInfNormBall(vector_var, radius=1.0)
        vector_var.value = torch.tensor([0.5, -0.5, 1.0])
        assert torch.allclose(atom.forward(), torch.tensor(0.0))

        vector_var.value = torch.tensor([0.5, -1.5, 0.0])
        assert torch.isinf(atom.forward())

    def test_is_smooth_and_proxable(self, vector_var):
        """LInfNormBall is nonsmooth and proxable for Variable input."""
        atom = LInfNormBall(vector_var, radius=1.0)
        assert atom.is_smooth() is False
        assert atom.is_proxable() is True

    def test_prox_clamps_to_bounds(self, vector_var):
        """Prox clamps values to [-radius, radius]."""
        atom = LInfNormBall(vector_var, radius=1.0)
        location = TensorDict({vector_var.name: torch.tensor([2.0, -3.0, 0.5])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.tensor([1.0, -1.0, 0.5])})
        assert_allclose_td(result, expected)


def test_negative_radius_raises(vector_var):
    """Negative radius should raise ValueError for all norm balls."""
    with pytest.raises(ValueError, match="radius must be non-negative"):
        L1NormBall(vector_var, radius=-1.0)
    with pytest.raises(ValueError, match="radius must be non-negative"):
        L2NormBall(vector_var, radius=-1.0)
    with pytest.raises(ValueError, match="radius must be non-negative"):
        LInfNormBall(vector_var, radius=-1.0)
