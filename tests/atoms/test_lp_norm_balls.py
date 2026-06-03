"""Tests for the Lp-norm ball constraint atoms."""

from abc import ABC, abstractmethod

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


class BaseNormBallTest(ABC):
    """Shared test suite for the Lp-norm ball constraint atoms.

    Concrete subclasses bind the atom class, a ball radius, feasible and
    infeasible points for that radius, and a (location, projection) pair, and
    inherit the shared forward / property / prox / decompose tests. The base
    class itself is not collected by pytest (it is not named ``Test*``).
    """

    @property
    @abstractmethod
    def atom_cls(self) -> type:
        """The norm ball atom class under test."""
        ...

    @property
    @abstractmethod
    def radius(self) -> float:
        """Ball radius used for the shared tests."""
        ...

    @property
    @abstractmethod
    def feasible_point(self) -> list[float]:
        """A point with norm <= radius."""
        ...

    @property
    @abstractmethod
    def infeasible_point(self) -> list[float]:
        """A point with norm > radius."""
        ...

    @property
    @abstractmethod
    def prox_location(self) -> list[float]:
        """An infeasible point to project onto the ball."""
        ...

    @property
    @abstractmethod
    def prox_expected(self) -> list[float]:
        """The projection of ``prox_location`` onto the ball of ``radius``."""
        ...

    def test_forward_satisfied_and_violated(self, vector_var):
        """Forward returns 0 within the ball and inf when violated."""
        atom = self.atom_cls(vector_var, radius=self.radius)
        vector_var.value = torch.tensor(self.feasible_point)
        assert torch.allclose(atom.forward(), torch.tensor(0.0))
        vector_var.value = torch.tensor(self.infeasible_point)
        assert torch.isinf(atom.forward())

    def test_is_smooth_and_proxable(self, vector_var):
        """Norm balls are nonsmooth and proxable for a Variable input."""
        atom = self.atom_cls(vector_var, radius=self.radius)
        assert atom.is_smooth() is False
        assert atom.is_proxable() is True

    def test_is_proxable_false_for_expression(self, vector_var):
        """Norm balls are not proxable when the input is not a Variable."""
        atom = self.atom_cls(torch.eye(3) @ vector_var + torch.ones(3), radius=1.0)
        assert atom.is_proxable() is False

    def test_prox_projects_onto_ball(self, vector_var):
        """Prox projects an infeasible point onto the ball."""
        atom = self.atom_cls(vector_var, radius=self.radius)
        location = TensorDict({vector_var.name: torch.tensor(self.prox_location)})
        result = atom.prox(location, prox_scaling=1.0)
        expected = TensorDict({vector_var.name: torch.tensor(self.prox_expected)})
        assert_allclose_td(result, expected)

    def test_prox_preserves_feasible_point(self, vector_var):
        """Prox leaves a feasible point unchanged."""
        atom = self.atom_cls(vector_var, radius=self.radius)
        location = TensorDict({vector_var.name: torch.tensor(self.feasible_point)})
        result = atom.prox(location, prox_scaling=1.0)
        assert_allclose_td(result, location)

    def test_decompose_affine_and_non_affine(self, vector_var):
        """Decompose works for affine inputs and returns None for non-affine."""
        affine_expr = torch.eye(3) @ vector_var + torch.ones(3)
        atom = self.atom_cls(affine_expr, radius=self.radius)
        decompositions = atom.decompose()
        assert decompositions is not None
        assert len(decompositions) == 1
        assert decompositions[0].affine_expr is affine_expr

        atom_non_affine = self.atom_cls(SumSquares(vector_var), radius=self.radius)
        assert atom_non_affine.decompose() is None

    def test_negative_radius_raises(self, vector_var):
        """A negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be non-negative"):
            self.atom_cls(vector_var, radius=-1.0)


class TestL1NormBall(BaseNormBallTest):
    """Tests for the L1NormBall atom."""

    atom_cls = L1NormBall
    radius = 2.0
    feasible_point = [1.0, -0.5, 0.0]  # L1 = 1.5
    infeasible_point = [2.0, 1.0, 0.0]  # L1 = 3.0
    prox_location = [3.0, -1.0, 0.0]  # L1 = 4.0
    prox_expected = [2.0, 0.0, 0.0]


class TestL2NormBall(BaseNormBallTest):
    """Tests for the L2NormBall atom."""

    atom_cls = L2NormBall
    radius = 2.0
    feasible_point = [1.0, 1.0, 0.0]  # L2 = sqrt(2)
    infeasible_point = [2.0, 2.0, 0.0]  # L2 = sqrt(8)
    prox_location = [3.0, 4.0, 0.0]  # L2 = 5.0
    prox_expected = [1.2, 1.6, 0.0]


class TestLInfNormBall(BaseNormBallTest):
    """Tests for the LInfNormBall atom."""

    atom_cls = LInfNormBall
    radius = 1.0
    feasible_point = [0.5, -0.5, 1.0]  # Linf = 1.0
    infeasible_point = [0.5, -1.5, 0.0]  # Linf = 1.5
    prox_location = [2.0, -3.0, 0.5]  # Linf = 3.0
    prox_expected = [1.0, -1.0, 0.5]
