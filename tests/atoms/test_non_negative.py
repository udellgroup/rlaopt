"""Tests for NonNegative atom."""

import pytest
import torch

from rlaopt.atoms.non_negative import NonNegative
from rlaopt.expression import Variable


@pytest.fixture
def scalar_nonneg():
    """Fixture for a scalar non-negative constraint."""
    x = Variable((1,), name="x")
    return NonNegative(x), x


@pytest.fixture
def vector_nonneg():
    """Fixture for a vector non-negative constraint."""
    x = Variable((5,), name="x")
    return NonNegative(x), x


@pytest.fixture
def matrix_nonneg():
    """Fixture for a matrix non-negative constraint."""
    x = Variable((3, 4), name="x")
    return NonNegative(x), x


class TestNonNegativeInit:
    """Tests for NonNegative initialization."""

    def test_scalar_initialization(self, scalar_nonneg):
        """Test initialization with scalar variable."""
        constraint, _ = scalar_nonneg

        assert constraint is not None
        assert constraint.is_proxable()

    def test_vector_initialization(self, vector_nonneg):
        """Test initialization with vector variable."""
        constraint, _ = vector_nonneg

        assert constraint is not None
        assert constraint.is_proxable()

    def test_matrix_initialization(self, matrix_nonneg):
        """Test initialization with matrix variable."""
        constraint, _ = matrix_nonneg

        assert constraint is not None
        assert constraint.is_proxable()

    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (10,),
            (100,),
            (5, 5),
            (3, 7),
            (2, 3, 4),
        ],
    )
    def test_various_shapes(self, shape):
        """Test initialization with various variable shapes."""
        x = Variable(shape, name="x")
        constraint = NonNegative(x)

        assert constraint is not None
        assert constraint.is_proxable()


class TestNonNegativeProx:
    """Tests for proximal operator (projection)."""

    def test_prox_all_positive(self, vector_nonneg):
        """Test projection when all values are already positive."""
        constraint, _ = vector_nonneg

        location = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        # Should remain unchanged
        assert torch.allclose(projected, location)
        assert torch.all(projected >= 0)

    def test_prox_all_negative(self, vector_nonneg):
        """Test projection when all values are negative."""
        constraint, _ = vector_nonneg

        location = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        # Should be clamped to zero
        expected = torch.zeros(5)
        assert torch.allclose(projected, expected)
        assert torch.all(projected >= 0)

    def test_prox_mixed_values(self, vector_nonneg):
        """Test projection with mixed positive and negative values."""
        constraint, _ = vector_nonneg

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        expected = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0])
        assert torch.allclose(projected, expected)
        assert torch.all(projected >= 0)

    def test_prox_zero_values(self, vector_nonneg):
        """Test projection with zero values."""
        constraint, _ = vector_nonneg

        location = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(projected, expected)
        assert torch.all(projected >= 0)

    @pytest.mark.parametrize(
        "location,expected",
        [
            (torch.tensor([5.0]), torch.tensor([5.0])),
            (torch.tensor([-5.0]), torch.tensor([0.0])),
            (torch.tensor([0.0]), torch.tensor([0.0])),
            (torch.tensor([1e-10]), torch.tensor([1e-10])),
            (torch.tensor([-1e-10]), torch.tensor([0.0])),
        ],
    )
    def test_prox_scalar_cases(self, scalar_nonneg, location, expected):
        """Test projection for various scalar cases."""
        constraint, _ = scalar_nonneg

        projected = constraint.prox(location, prox_scaling=1.0)

        assert torch.allclose(projected, expected)
        assert torch.all(projected >= 0)

    def test_prox_matrix(self, matrix_nonneg):
        """Test projection with matrix-shaped variable."""
        constraint, _ = matrix_nonneg

        location = torch.tensor(
            [[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0], [9.0, -10.0, 11.0, -12.0]]
        )
        projected = constraint.prox(location, prox_scaling=1.0)

        expected = torch.tensor(
            [[1.0, 0.0, 3.0, 0.0], [0.0, 6.0, 0.0, 8.0], [9.0, 0.0, 11.0, 0.0]]
        )
        assert torch.allclose(projected, expected)
        assert torch.all(projected >= 0)

    @pytest.mark.parametrize("prox_scaling", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_prox_scaling_independence(self, vector_nonneg, prox_scaling):
        """Test that projection is independent of prox_scaling for hard constraints."""
        constraint, _ = vector_nonneg

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])

        proj_baseline = constraint.prox(location, prox_scaling=1.0)
        proj_scaled = constraint.prox(location, prox_scaling=prox_scaling)

        # Box constraints are independent of scaling
        assert torch.allclose(proj_baseline, proj_scaled)

    def test_prox_very_negative(self, vector_nonneg):
        """Test projection with very large negative values."""
        constraint, _ = vector_nonneg

        location = torch.tensor([-1e6, -1e9, -1e3, 1.0, -1e12])
        projected = constraint.prox(location, prox_scaling=1.0)

        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        assert torch.allclose(projected, expected)
        assert torch.all(projected >= 0)

    def test_prox_boundary_values(self, vector_nonneg):
        """Test projection with values very close to zero boundary."""
        constraint, _ = vector_nonneg

        location = torch.tensor([1e-8, -1e-8, 1e-7, -1e-7, 0.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        # Positive values stay, negative become zero
        assert projected[0] > 0
        assert projected[1] == 0
        assert projected[2] > 0
        assert projected[3] == 0
        assert projected[4] == 0
        assert torch.all(projected >= 0)


class TestNonNegativeProperties:
    """Tests for constraint properties."""

    def test_is_proxable(self, vector_nonneg):
        """Test that is_proxable returns True."""
        constraint, _ = vector_nonneg

        assert constraint.is_proxable() is True

    def test_idempotency(self, vector_nonneg):
        """Test that projecting twice gives the same result."""
        constraint, _ = vector_nonneg

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        projected_once = constraint.prox(location, prox_scaling=1.0)
        projected_twice = constraint.prox(projected_once, prox_scaling=1.0)

        assert torch.allclose(projected_once, projected_twice)

    @pytest.mark.parametrize("seed", range(5))
    def test_idempotency_random(self, vector_nonneg, seed):
        """Test idempotency with random values."""
        constraint, _ = vector_nonneg

        torch.manual_seed(seed)
        location = torch.randn(5)
        projected_once = constraint.prox(location, prox_scaling=1.0)
        projected_twice = constraint.prox(projected_once, prox_scaling=1.0)

        assert torch.allclose(projected_once, projected_twice)
        assert torch.all(projected_once >= 0)

    def test_projection_maintains_shape(self, matrix_nonneg):
        """Test that projection maintains the variable shape."""
        constraint, _ = matrix_nonneg

        location = torch.randn(3, 4)
        projected = constraint.prox(location, prox_scaling=1.0)

        assert projected.shape == location.shape

    def test_projection_element_wise(self, vector_nonneg):
        """Test that projection operates element-wise."""
        constraint, _ = vector_nonneg

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        # Each element should be max(location[i], 0)
        for i in range(len(location)):
            expected_i = max(location[i].item(), 0.0)
            assert torch.allclose(projected[i], torch.tensor(expected_i))


class TestNonNegativeEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point precisions."""
        x = Variable((5,), name="x")
        constraint = NonNegative(x)

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=dtype)
        projected = constraint.prox(location, prox_scaling=1.0)

        expected = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0], dtype=dtype)
        assert torch.allclose(projected, expected)
        assert projected.dtype == dtype
        assert torch.all(projected >= 0)

    def test_large_scale(self):
        """Test with large-scale variables."""
        x = Variable((1000,), name="x")
        constraint = NonNegative(x)

        location = torch.randn(1000)
        projected = constraint.prox(location, prox_scaling=1.0)

        assert projected.shape == location.shape
        assert torch.all(projected >= 0)
        # Check that positive values are preserved
        pos_mask = location > 0
        assert torch.allclose(projected[pos_mask], location[pos_mask])
        # Check that negative values are zeroed
        neg_mask = location < 0
        assert torch.allclose(
            projected[neg_mask], torch.zeros_like(projected[neg_mask])
        )

    def test_very_small_positive(self):
        """Test with very small positive values near machine epsilon."""
        x = Variable((3,), name="x")
        constraint = NonNegative(x)

        location = torch.tensor([1e-20, 1e-15, 1e-10], dtype=torch.float64)
        projected = constraint.prox(location, prox_scaling=1.0)

        # Should preserve small positive values
        assert torch.allclose(projected, location)
        assert torch.all(projected >= 0)

    def test_infinity_values(self):
        """Test with infinite values."""
        x = Variable((4,), name="x")
        constraint = NonNegative(x)

        location = torch.tensor([float("inf"), float("-inf"), 1.0, -1.0])
        projected = constraint.prox(location, prox_scaling=1.0)

        assert projected[0] == float("inf")
        assert projected[1] == 0.0
        assert projected[2] == 1.0
        assert projected[3] == 0.0

    def test_3d_tensor(self):
        """Test with 3D tensor variable."""
        x = Variable((2, 3, 4), name="x")
        constraint = NonNegative(x)

        location = torch.randn(2, 3, 4)
        projected = constraint.prox(location, prox_scaling=1.0)

        assert projected.shape == location.shape
        assert torch.all(projected >= 0)
        # Verify element-wise max(x, 0) behavior
        expected = torch.maximum(location, torch.zeros_like(location))
        assert torch.allclose(projected, expected)


class TestNonNegativeExamples:
    """Tests based on docstring examples."""

    def test_docstring_basic_example(self):
        """Test the basic example from docstring."""
        x = Variable((10,), name="x")
        nonneg = NonNegative(x)

        assert nonneg is not None
        assert nonneg.is_proxable()

    def test_docstring_projection_example(self):
        """Test the projection example from docstring."""
        x = Variable((10,), name="x")
        nonneg = NonNegative(x)

        torch.manual_seed(42)
        point_with_negatives = torch.randn(10)
        projected = nonneg.prox(point_with_negatives, prox_scaling=1.0)

        # All negative values should be clamped to 0
        assert torch.all(projected >= 0)
        # Positive values should be preserved
        pos_mask = point_with_negatives > 0
        assert torch.allclose(projected[pos_mask], point_with_negatives[pos_mask])


class TestNonNegativeComparison:
    """Tests comparing NonNegative with equivalent Box constraints."""

    def test_equivalent_to_box(self):
        """Test that NonNegative is equivalent to Box(lower=0, upper=None)."""
        from rlaopt.atoms.box import Box

        x1 = Variable((5,), name="x1")
        x2 = Variable((5,), name="x2")

        nonneg = NonNegative(x1)
        box = Box(x2, lower=0.0, upper=None)

        location = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])

        proj_nonneg = nonneg.prox(location, prox_scaling=1.0)
        proj_box = box.prox(location, prox_scaling=1.0)

        assert torch.allclose(proj_nonneg, proj_box)


@pytest.mark.parametrize(
    "shape,seed",
    [
        ((10,), 0),
        ((5, 5), 1),
        ((3, 4, 5), 2),
        ((100,), 3),
    ],
)
def test_nonnegative_general(shape, seed):
    """General test for NonNegative with various configurations."""
    torch.manual_seed(seed)

    x = Variable(shape, name="x")
    constraint = NonNegative(x)

    location = torch.randn(shape)
    projected = constraint.prox(location, prox_scaling=1.0)

    # Verify non-negativity
    assert torch.all(projected >= 0)

    # Verify idempotency
    projected_twice = constraint.prox(projected, prox_scaling=1.0)
    assert torch.allclose(projected, projected_twice)

    # Verify shape preservation
    assert projected.shape == location.shape

    # Verify element-wise behavior: max(x, 0)
    expected = torch.maximum(location, torch.zeros_like(location))
    assert torch.allclose(projected, expected)
