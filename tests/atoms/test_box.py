"""Tests for Box atom."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms.box import Box
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict


@pytest.fixture
def vector_var():
    """Create a vector variable."""
    return Variable((5,), name="x")


@pytest.fixture
def standard_box(vector_var):
    """Create a standard box constraint: 0 <= x <= 1."""
    return Box(vector_var, lower=0.0, upper=1.0)


@pytest.fixture
def nonneg_box(vector_var):
    """Create a non-negativity constraint: x >= 0."""
    return Box(vector_var, lower=0.0)


@pytest.fixture
def upper_only_box(vector_var):
    """Create an upper bound constraint: x <= 1."""
    return Box(vector_var, upper=1.0)


class TestBox:
    """Test Box constraint atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_both_bounds(self, vector_var):
        """Test initialization with both lower and upper bounds."""
        box = Box(vector_var, lower=torch.zeros(5), upper=torch.ones(5))
        assert box.get_buffer("lower") is not None
        assert box.get_buffer("upper") is not None
        assert box.get_buffer("A") is None
        assert box.get_buffer("b") is None
        assert box.get_buffer("C") is None

    def test_init_with_scalar_bounds(self, vector_var):
        """Test initialization with scalar bounds."""
        box = Box(vector_var, lower=0.0, upper=1.0)
        assert box.get_buffer("lower").shape == torch.Size([])
        assert box.get_buffer("upper").shape == torch.Size([])

    def test_init_with_only_lower_bound(self, nonneg_box):
        """Test initialization with only lower bound sets upper to infinity."""
        assert nonneg_box.get_buffer("lower") is not None
        assert (
            torch.isinf(nonneg_box.get_buffer("upper"))
            and nonneg_box.get_buffer("upper") > 0
        )

    def test_init_with_only_upper_bound(self, upper_only_box):
        """Test initialization with only upper bound sets lower to -infinity."""
        assert upper_only_box.get_buffer("upper") is not None
        assert (
            torch.isinf(upper_only_box.get_buffer("lower"))
            and upper_only_box.get_buffer("lower") < 0
        )

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_returns_zero_when_satisfied(self, standard_box, vector_var):
        """Test forward() returns 0 when constraints are satisfied."""
        vector_var.value = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        result = standard_box.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    def test_forward_returns_inf_when_below_lower(self, standard_box, vector_var):
        """Test forward() returns infinity when below lower bound."""
        vector_var.value = torch.tensor([-0.1, 0.5, 0.5, 0.5, 0.5])
        result = standard_box.forward()
        assert torch.isinf(result)

    def test_forward_returns_inf_when_above_upper(self, standard_box, vector_var):
        """Test forward() returns infinity when above upper bound."""
        vector_var.value = torch.tensor([0.5, 1.1, 0.5, 0.5, 0.5])
        result = standard_box.forward()
        assert torch.isinf(result)

    def test_forward_satisfied_at_boundary(self, standard_box, vector_var):
        """Test forward() returns 0 when exactly at bounds."""
        vector_var.value = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0])
        result = standard_box.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_false(self, standard_box):
        """Test box constraint is not smooth (inherits from Polyhedron)."""
        assert standard_box.is_smooth() is False

    def test_is_proxable_returns_true(self, standard_box):
        """Test box constraint is proxable."""
        assert standard_box.is_proxable() is True

    # ----------------------
    # Proximal operator tests
    # ----------------------

    def test_prox_clamps_values_within_bounds(self, standard_box, vector_var):
        """Test prox() clamps values to lie within bounds."""
        location = TensorDict(
            {vector_var.name: torch.tensor([-1.0, 0.5, 2.0, 0.0, 1.5])}
        )
        result = standard_box.prox(location, prox_scaling=1.0)
        expected = TensorDict(
            {vector_var.name: torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0])}
        )
        assert_allclose_td(result, expected)

    def test_prox_is_independent_of_scaling(self, standard_box, vector_var):
        """Test prox() gives same result regardless of scaling parameter."""
        location = TensorDict(
            {vector_var.name: torch.tensor([-1.0, 0.5, 2.0, 0.0, 1.5])}
        )
        result1 = standard_box.prox(location, prox_scaling=1.0)
        result2 = standard_box.prox(location, prox_scaling=100.0)
        assert_allclose_td(result1, result2)

    def test_prox_with_lower_bound_only(self, nonneg_box, vector_var):
        """Test prox() with only lower bound enforces non-negativity."""
        location = TensorDict(
            {vector_var.name: torch.tensor([-5.0, 0.5, 10.0, -1.0, 3.0])}
        )
        result = nonneg_box.prox(location, prox_scaling=1.0)
        # Should clamp negative values to 0, leave positive unchanged
        expected = TensorDict(
            {vector_var.name: torch.tensor([0.0, 0.5, 10.0, 0.0, 3.0])}
        )
        assert_allclose_td(result, expected)

    def test_prox_with_upper_bound_only(self, upper_only_box, vector_var):
        """Test prox() with only upper bound."""
        location = TensorDict(
            {vector_var.name: torch.tensor([-5.0, 0.5, 10.0, -1.0, 0.8])}
        )
        result = upper_only_box.prox(location, prox_scaling=1.0)
        # Should clamp values above 1 to 1, leave others unchanged
        expected = TensorDict(
            {vector_var.name: torch.tensor([-5.0, 0.5, 1.0, -1.0, 0.8])}
        )
        assert_allclose_td(result, expected)

    def test_prox_returns_identity_for_feasible_point(self, standard_box, vector_var):
        """Test prox() returns input unchanged if already feasible."""
        location = TensorDict(
            {vector_var.name: torch.tensor([0.3, 0.5, 0.7, 0.2, 0.9])}
        )
        result = standard_box.prox(location, prox_scaling=1.0)
        assert_allclose_td(result, location)

    # ----------------------
    # Edge cases
    # ----------------------

    def test_with_vector_bounds(self, vector_var):
        """Test box constraint with different bounds per dimension."""
        lower = torch.tensor([0.0, -1.0, 0.5, -2.0, 1.0])
        upper = torch.tensor([1.0, 1.0, 2.0, 0.0, 3.0])
        box = Box(vector_var, lower=lower, upper=upper)

        location = TensorDict(
            {vector_var.name: torch.tensor([-1.0, 0.0, 1.0, -3.0, 5.0])}
        )
        result = box.prox(location, prox_scaling=1.0)
        expected = TensorDict(
            {vector_var.name: torch.tensor([0.0, 0.0, 1.0, -2.0, 3.0])}
        )
        assert_allclose_td(result, expected)

    def test_unbounded_box_allows_any_value(self, vector_var):
        """Test box with no bounds (unconstrained), leads to a ValueError."""
        with pytest.raises(ValueError, match="Provided constraints"):
            Box(vector_var)  # No bounds specified
