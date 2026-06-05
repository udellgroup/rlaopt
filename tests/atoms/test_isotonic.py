"""Tests for the isotonic constraint atom."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import Isotonic, SumSquares
from rlaopt.atoms.isotonic import _prox_isotonic
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((4,), name="x")


class TestIsotonic:
    """Tests for Isotonic atom."""

    def test_forward_satisfied_and_violated(self, vector_var):
        """Forward returns 0 if non-decreasing and inf otherwise."""
        atom = Isotonic(vector_var)
        vector_var.value = torch.tensor([1.0, 1.0, 2.0, 3.0])
        assert torch.allclose(atom.forward(), torch.tensor(0.0))

        vector_var.value = torch.tensor([1.0, 3.0, 2.0, 4.0])
        assert torch.isinf(atom.forward())

    def test_is_smooth_and_proxable(self, vector_var):
        """Isotonic is nonsmooth and proxable for Variable input."""
        atom = Isotonic(vector_var)
        assert atom.is_smooth() is False
        assert atom.is_proxable() is True

    def test_is_proxable_false_for_expression(self, vector_var):
        """Isotonic is not proxable when input is not a Variable."""
        A = torch.eye(4)
        b = torch.ones(4)
        atom = Isotonic(A @ vector_var + b)
        assert atom.is_proxable() is False

    def test_prox_projects_onto_cone(self, vector_var):
        """Prox projects a fully-decreasing point to the overall mean."""
        atom = Isotonic(vector_var)
        location = TensorDict({vector_var.name: torch.tensor([3.0, 2.0, 1.0, 0.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.tensor([1.5, 1.5, 1.5, 1.5])})
        assert_allclose_td(result, expected)

    def test_prox_matches_pava_reference(self, vector_var):
        """Prox matches a hand-computed PAVA result with a pooled middle block."""
        atom = Isotonic(vector_var)
        location = TensorDict({vector_var.name: torch.tensor([1.0, 3.0, 2.0, 4.0])})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.tensor([1.0, 2.5, 2.5, 4.0])})
        assert_allclose_td(result, expected)

    def test_prox_preserves_feasible_point(self, vector_var):
        """Prox leaves an already non-decreasing point unchanged."""
        atom = Isotonic(vector_var)
        location = TensorDict({vector_var.name: torch.tensor([1.0, 2.0, 2.0, 5.0])})
        result = atom.prox(location, prox_scaling=1.0)
        assert_allclose_td(result, location)

    def test_prox_idempotent(self, vector_var):
        """Projecting twice equals projecting once."""
        atom = Isotonic(vector_var)
        location = TensorDict({vector_var.name: torch.tensor([4.0, 1.0, 3.0, 2.0])})
        once = atom.prox(location, prox_scaling=1.0)
        twice = atom.prox(once, prox_scaling=1.0)
        assert_allclose_td(twice, once)

    def test_decompose_affine_and_non_affine(self, vector_var):
        """Decompose works for affine inputs and returns None for non-affine."""
        A = torch.eye(4)
        b = torch.ones(4)
        affine_expr = A @ vector_var + b
        atom = Isotonic(affine_expr)
        decompositions = atom.decompose()
        assert decompositions is not None
        assert len(decompositions) == 1
        assert decompositions[0].affine_expr is affine_expr

        non_affine = SumSquares(vector_var)
        atom_non_affine = Isotonic(non_affine)
        assert atom_non_affine.decompose() is None

    def test_scaling_returns_same_atom(self, vector_var):
        """Scaling an isotonic constraint returns the same atom (identity)."""
        atom = Isotonic(vector_var)
        scaled = 5.0 * atom
        assert scaled is atom

    def test_scaling_by_zero_warns(self, vector_var):
        """Scaling by zero raises a warning but still returns the same atom."""
        atom = Isotonic(vector_var)
        with pytest.warns(UserWarning, match="Scaling a Isotonic.*by zero"):
            scaled = 0.0 * atom
        assert scaled is atom

    def test_prox_gradient_flows_through(self, vector_var):
        """Prox allows gradient computation without NaNs."""
        atom = Isotonic(vector_var)
        location_tensor = torch.tensor([3.0, 1.0, 5.0, 2.0], requires_grad=True)
        location = TensorDict({vector_var.name: location_tensor})

        result = atom.prox(location, prox_scaling=1.0)
        result[vector_var.name].sum().backward()

        assert location_tensor.grad is not None
        assert not torch.isnan(location_tensor.grad).any()

    def test_prox_gradcheck(self):
        """Gradcheck the projection at an interior point of a fixed-block region."""
        # Blocks {0,1}, {2,3}, {4} with distinct means (2.0, 3.5, 8.0), so the
        # partition is locally stable and the projection is locally linear.
        x = torch.tensor(
            [3.0, 1.0, 5.0, 2.0, 8.0], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(_prox_isotonic.apply, (x,))
