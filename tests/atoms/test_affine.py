"""Tests for Affine atom."""

import pytest
import torch

from rlaopt.atoms.affine import Affine
from rlaopt.expression import Variable


@pytest.fixture
def simple_affine():
    """Fixture for a simple 2D affine transformation."""
    x = Variable((2,), name="x")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([5.0, 6.0])
    x.value = torch.tensor([1.0, 1.0])
    return Affine(x, A, b), x, A, b


@pytest.fixture
def vector_affine():
    """Fixture for a vector affine transformation."""
    x = Variable((5,), name="x")
    A = torch.randn(3, 5)
    b = torch.randn(3)
    x.value = torch.randn(5)
    return Affine(x, A, b), x, A, b


@pytest.fixture
def identity_affine():
    """Fixture for identity transformation (A = I, b = 0)."""
    x = Variable((4,), name="x")
    A = torch.eye(4)
    b = torch.zeros(4)
    x.value = torch.randn(4)
    return Affine(x, A, b), x, A, b


class TestAffineInit:
    """Tests for Affine initialization."""

    def test_basic_initialization(self, simple_affine):
        """Test basic initialization with valid inputs."""
        affine, x, A, b = simple_affine

        assert affine is not None
        assert torch.allclose(affine.A, A)
        assert torch.allclose(affine.b, b)

    def test_initialization_registers_variable(self, simple_affine):
        """Test that initialization registers the variable."""
        affine, x, _, _ = simple_affine

        assert hasattr(affine, "var_name")
        assert affine.var_name == x.name

    @pytest.mark.parametrize(
        "m,n",
        [
            (1, 1),
            (3, 5),
            (10, 20),
            (5, 5),  # Square
        ],
    )
    def test_various_dimensions(self, m, n):
        """Test initialization with various matrix dimensions."""
        x = Variable((n,), name="x")
        A = torch.randn(m, n)
        b = torch.randn(m)

        affine = Affine(x, A, b)

        assert affine is not None
        assert affine.A.shape == (m, n)
        assert affine.b.shape == (m,)

    def test_initialization_with_non_variable_raises_error(self):
        """Test that initialization with non-Variable raises TypeError."""
        A = torch.randn(3, 5)
        b = torch.randn(3)
        not_a_variable = torch.randn(5)

        with pytest.raises(TypeError, match="Expected Variable"):
            Affine(not_a_variable, A, b)

    def test_initialization_with_none_raises_error(self):
        """Test that initialization with None raises TypeError."""
        A = torch.randn(3, 5)
        b = torch.randn(3)

        with pytest.raises(TypeError, match="Expected Variable"):
            Affine(None, A, b)

    def test_buffers_registered(self, simple_affine):
        """Test that A and b are registered as buffers."""
        affine, _, A, b = simple_affine

        assert hasattr(affine, "A")
        assert hasattr(affine, "b")
        assert isinstance(affine.A, torch.Tensor)
        assert isinstance(affine.b, torch.Tensor)


class TestAffineForward:
    """Tests for forward evaluation."""

    def test_forward_simple(self, simple_affine):
        """Test forward with simple known values."""
        affine, x, A, b = simple_affine

        # x = [1, 1], A @ x = [3, 7], A @ x + b = [8, 13]
        result = affine.forward()
        expected = torch.tensor([8.0, 13.0])

        assert torch.allclose(result, expected)

    def test_forward_identity(self, identity_affine):
        """Test forward with identity transformation."""
        affine, x, _, _ = identity_affine

        result = affine.forward()

        # Should return x unchanged (A = I, b = 0)
        assert torch.allclose(result, x.value)

    def test_forward_zero_matrix(self):
        """Test forward with zero matrix (output = b)."""
        x = Variable((5,), name="x")
        A = torch.zeros(3, 5)
        b = torch.tensor([1.0, 2.0, 3.0])
        x.value = torch.randn(5)

        affine = Affine(x, A, b)
        result = affine.forward()

        # A @ x = 0, so result = b
        assert torch.allclose(result, b)

    def test_forward_zero_bias(self):
        """Test forward with zero bias (output = A @ x)."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.zeros(1)
        x.value = torch.tensor([1.0, 1.0, 1.0])

        affine = Affine(x, A, b)
        result = affine.forward()

        # Result = A @ x = [6.0]
        expected = torch.tensor([6.0])
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("seed", range(5))
    def test_forward_random(self, seed):
        """Test forward with random matrices and values."""
        torch.manual_seed(seed)

        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        x.value = torch.randn(5)

        affine = Affine(x, A, b)
        result = affine.forward()

        # Manually compute expected result
        expected = A @ x.value + b
        assert torch.allclose(result, expected)

    def test_forward_uses_current_variable_value(self):
        """Test that forward uses the variable value at initialization."""
        x = Variable((2,), name="x")
        A = torch.eye(2)
        b = torch.zeros(2)
        x.value = torch.tensor([1.0, 2.0])

        affine = Affine(x, A, b)
        result = affine.forward()

        expected = A @ x.value + b
        assert torch.allclose(result, expected)

    def test_forward_large_scale(self):
        """Test forward with large-scale transformation."""
        x = Variable((100,), name="x")
        A = torch.randn(50, 100)
        b = torch.randn(50)
        x.value = torch.randn(100)

        affine = Affine(x, A, b)
        result = affine.forward()

        expected = A @ x.value + b
        assert torch.allclose(result, expected)
        assert result.shape == (50,)


class TestAffineProperties:
    """Tests for affine properties."""

    def test_is_smooth_always_true(self, simple_affine):
        """Test that is_smooth always returns True."""
        affine, _, _, _ = simple_affine

        assert affine.is_smooth() is True

    def test_is_proxable_always_false(self, simple_affine):
        """Test that is_proxable always returns False."""
        affine, _, _, _ = simple_affine

        assert affine.is_proxable() is False

    def test_is_subsamplable_always_true(self, simple_affine):
        """Test that is_subsamplable always returns True."""
        affine, _, _, _ = simple_affine

        assert affine.is_subsamplable() is True

    @pytest.mark.parametrize(
        "m,n",
        [
            (5, 3),
            (10, 10),
            (20, 15),
        ],
    )
    def test_properties_consistent(self, m, n):
        """Test that properties are consistent across different dimensions."""
        x = Variable((n,), name="x")
        A = torch.randn(m, n)
        b = torch.randn(m)
        affine = Affine(x, A, b)

        assert affine.is_smooth() is True
        assert affine.is_proxable() is False
        assert affine.is_subsamplable() is True


class TestAffineProx:
    """Tests for prox operator (should raise error)."""

    def test_prox_raises_not_implemented(self, simple_affine):
        """Test that prox raises NotImplementedError."""
        affine, _, _, _ = simple_affine

        with pytest.raises(NotImplementedError, match="Affine is not proxable"):
            affine.prox(torch.randn(2), prox_scaling=1.0)

    @pytest.mark.parametrize("prox_scaling", [0.1, 1.0, 10.0])
    def test_prox_raises_regardless_of_scaling(self, simple_affine, prox_scaling):
        """Test that prox raises error regardless of scaling parameter."""
        affine, _, _, _ = simple_affine

        with pytest.raises(NotImplementedError):
            affine.prox(torch.randn(2), prox_scaling=prox_scaling)


class TestAffineSubsample:
    """Tests for subsampling functionality."""

    def test_subsample_single_index(self):
        """Test subsampling with a single index."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        affine = Affine(x, A, b)

        indices = torch.tensor([3])
        sub_affine = affine.subsample(indices)

        assert sub_affine.A.shape == (1, 5)
        assert sub_affine.b.shape == (1,)
        assert torch.allclose(sub_affine.A, A[indices])
        assert torch.allclose(sub_affine.b, b[indices])

    def test_subsample_multiple_indices(self):
        """Test subsampling with multiple indices."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        x.value = torch.randn(5)
        affine = Affine(x, A, b)

        indices = torch.tensor([0, 2, 5, 7])
        sub_affine = affine.subsample(indices)

        assert sub_affine.A.shape == (4, 5)
        assert sub_affine.b.shape == (4,)
        assert torch.allclose(sub_affine.A, A[indices])
        assert torch.allclose(sub_affine.b, b[indices])

    def test_subsample_preserves_functionality(self):
        """Test that subsampled affine still computes correctly."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        x.value = torch.randn(5)
        affine = Affine(x, A, b)

        indices = torch.tensor([1, 3, 5])
        sub_affine = affine.subsample(indices)

        result = sub_affine.forward()
        expected = A[indices] @ x.value + b[indices]

        assert torch.allclose(result, expected)

    def test_subsample_all_indices(self):
        """Test subsampling with all indices (should be equivalent)."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        x.value = torch.randn(5)
        affine = Affine(x, A, b)

        indices = torch.arange(10)
        sub_affine = affine.subsample(indices)

        result_original = affine.forward()
        result_subsampled = sub_affine.forward()

        assert torch.allclose(result_original, result_subsampled)

    def test_subsample_reverse_order(self):
        """Test subsampling with indices in reverse order."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        x.value = torch.randn(5)
        affine = Affine(x, A, b)

        indices = torch.tensor([9, 7, 5, 3, 1])
        sub_affine = affine.subsample(indices)

        result = sub_affine.forward()
        expected = A[indices] @ x.value + b[indices]

        assert torch.allclose(result, expected)

    def test_subsample_duplicate_indices(self):
        """Test subsampling with duplicate indices."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        x.value = torch.randn(5)
        affine = Affine(x, A, b)

        indices = torch.tensor([2, 2, 5, 5])
        sub_affine = affine.subsample(indices)

        assert sub_affine.A.shape == (4, 5)
        # Each duplicated index appears twice
        assert torch.allclose(sub_affine.A[0], sub_affine.A[1])
        assert torch.allclose(sub_affine.A[2], sub_affine.A[3])

    def test_subsample_preserves_properties(self):
        """Test that subsampling preserves atom properties."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        affine = Affine(x, A, b)

        indices = torch.tensor([0, 3, 7])
        sub_affine = affine.subsample(indices)

        assert sub_affine.is_smooth() is True
        assert sub_affine.is_proxable() is False
        assert sub_affine.is_subsamplable() is True

    @pytest.mark.parametrize(
        "n_rows,n_samples",
        [
            (10, 3),
            (20, 10),
            (50, 25),
        ],
    )
    def test_subsample_various_sizes(self, n_rows, n_samples):
        """Test subsampling with various sizes."""
        x = Variable((5,), name="x")
        A = torch.randn(n_rows, 5)
        b = torch.randn(n_rows)
        x.value = torch.randn(5)
        affine = Affine(x, A, b)

        indices = torch.randperm(n_rows)[:n_samples]
        sub_affine = affine.subsample(indices)

        assert sub_affine.A.shape == (n_samples, 5)
        assert sub_affine.b.shape == (n_samples,)


class TestAffineEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point precisions."""
        x = Variable((3,), name="x")
        A = torch.randn(2, 3, dtype=dtype)
        b = torch.randn(2, dtype=dtype)
        x.value = torch.randn(3, dtype=dtype)

        affine = Affine(x, A, b)
        result = affine.forward()

        assert result.dtype == dtype
        expected = A @ x.value + b
        assert torch.allclose(result, expected)

    def test_very_small_values(self):
        """Test with very small values."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1e-10, 2e-10, 3e-10]], dtype=torch.float64)
        b = torch.tensor([1e-10], dtype=torch.float64)
        x.value = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        affine = Affine(x, A, b)
        result = affine.forward()

        expected = A @ x.value + b
        assert torch.allclose(result, expected, atol=1e-15)

    def test_very_large_values(self):
        """Test with very large values."""
        x = Variable((3,), name="x")
        A = torch.tensor([[1e6, 2e6, 3e6]])
        b = torch.tensor([1e6])
        x.value = torch.tensor([1.0, 1.0, 1.0])

        affine = Affine(x, A, b)
        result = affine.forward()

        expected = A @ x.value + b
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_rectangular_matrix_tall(self):
        """Test with tall rectangular matrix (m > n)."""
        x = Variable((5,), name="x")
        A = torch.randn(20, 5)
        b = torch.randn(20)
        x.value = torch.randn(5)

        affine = Affine(x, A, b)
        result = affine.forward()

        assert result.shape == (20,)
        expected = A @ x.value + b
        assert torch.allclose(result, expected)

    def test_rectangular_matrix_wide(self):
        """Test with wide rectangular matrix (m < n)."""
        x = Variable((20,), name="x")
        A = torch.randn(5, 20)
        b = torch.randn(5)
        x.value = torch.randn(20)

        affine = Affine(x, A, b)
        result = affine.forward()

        assert result.shape == (5,)
        expected = A @ x.value + b
        assert torch.allclose(result, expected)


class TestAffineExamples:
    """Tests based on docstring examples."""

    def test_docstring_basic_example(self):
        """Test the basic example from docstring."""
        x = Variable((5,), name="x")
        A = torch.randn(3, 5)
        b = torch.randn(3)
        x.value = torch.randn(5)

        affine = Affine(x, A, b)
        result = affine.forward()

        # Should compute A @ x + b
        expected = A @ x.value + b
        assert torch.allclose(result, expected)

    def test_docstring_subsample_example(self):
        """Test the subsample example from docstring."""
        x = Variable((5,), name="x")
        A = torch.randn(10, 5)
        b = torch.randn(10)
        affine = Affine(x, A, b)

        sub_affine = affine.subsample(torch.tensor([0, 2, 5]))

        assert sub_affine.A.shape == torch.Size([3, 5])


@pytest.mark.parametrize(
    "m,n,seed",
    [
        (3, 5, 0),
        (10, 8, 1),
        (5, 5, 2),
        (20, 10, 3),
    ],
)
def test_affine_general(m, n, seed):
    """General test for Affine with various configurations."""
    torch.manual_seed(seed)

    x = Variable((n,), name="x")
    A = torch.randn(m, n)
    b = torch.randn(m)
    x.value = torch.randn(n)

    affine = Affine(x, A, b)

    # Test forward
    result = affine.forward()
    expected = A @ x.value + b
    assert torch.allclose(result, expected)

    # Test properties
    assert affine.is_smooth()
    assert not affine.is_proxable()
    assert affine.is_subsamplable()

    # Test subsample
    n_samples = min(m, 5)
    indices = torch.randperm(m)[:n_samples]
    sub_affine = affine.subsample(indices)
    sub_result = sub_affine.forward()
    sub_expected = A[indices] @ x.value + b[indices]
    assert torch.allclose(sub_result, sub_expected)
