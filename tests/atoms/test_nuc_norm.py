"""Tests for NucNorm atom."""

import pytest
import torch

from rlaopt.atoms.nuc_norm import NucNorm
from rlaopt.expression import Variable


@pytest.fixture
def square_matrix():
    """Fixture for square matrix variable."""
    X = Variable((3, 3), name="X")
    X.value = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    return NucNorm(X, scaling=1.0), X


@pytest.fixture
def tall_matrix():
    """Fixture for tall rectangular matrix variable."""
    X = Variable((5, 3), name="X")
    X.value = torch.randn(5, 3)
    return NucNorm(X, scaling=1.0), X


@pytest.fixture
def wide_matrix():
    """Fixture for wide rectangular matrix variable."""
    X = Variable((3, 5), name="X")
    X.value = torch.randn(3, 5)
    return NucNorm(X, scaling=1.0), X


class TestNucNormInit:
    """Tests for NucNorm initialization."""

    def test_basic_initialization(self, square_matrix):
        """Test basic initialization with square matrix."""
        nuc_norm, X = square_matrix

        assert nuc_norm is not None
        assert nuc_norm.scaling == 1.0

    def test_initialization_with_custom_scaling(self):
        """Test initialization with custom scaling factor."""
        X = Variable((4, 4), name="X")
        X.value = torch.eye(4)
        nuc_norm = NucNorm(X, scaling=0.5)

        assert nuc_norm.scaling == 0.5

    @pytest.mark.parametrize("scaling", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_various_scaling_factors(self, scaling):
        """Test initialization with various scaling factors."""
        X = Variable((3, 3), name="X")
        X.value = torch.eye(3)
        nuc_norm = NucNorm(X, scaling=scaling)

        assert nuc_norm.scaling == scaling

    def test_initialization_with_tall_matrix(self, tall_matrix):
        """Test initialization with tall rectangular matrix."""
        nuc_norm, X = tall_matrix

        assert nuc_norm is not None
        assert X.value.shape == (5, 3)

    def test_initialization_with_wide_matrix(self, wide_matrix):
        """Test initialization with wide rectangular matrix."""
        nuc_norm, X = wide_matrix

        assert nuc_norm is not None
        assert X.value.shape == (3, 5)

    def test_initialization_with_non_variable_raises_error(self):
        """Test that initialization with non-Variable raises TypeError."""
        not_a_variable = torch.randn(3, 3)

        with pytest.raises(TypeError, match="Expected Variable"):
            NucNorm(not_a_variable, scaling=1.0)

    def test_initialization_with_1d_tensor_raises_error(self):
        """Test that initialization with 1D tensor raises ValueError."""
        X = Variable((5,), name="X")
        X.value = torch.randn(5)

        with pytest.raises(ValueError, match="must be 2D Tensor"):
            NucNorm(X, scaling=1.0)

    def test_initialization_with_3d_tensor_raises_error(self):
        """Test that initialization with 3D tensor raises ValueError."""
        X = Variable((2, 3, 4), name="X")
        X.value = torch.randn(2, 3, 4)

        with pytest.raises(ValueError, match="must be 2D Tensor"):
            NucNorm(X, scaling=1.0)

    def test_default_scaling(self):
        """Test that default scaling factor is 1.0."""
        X = Variable((3, 3), name="X")
        X.value = torch.eye(3)
        nuc_norm = NucNorm(X)

        assert nuc_norm.scaling == 1.0

    @pytest.mark.parametrize(
        "m,n",
        [
            (2, 2),
            (5, 5),
            (10, 10),
            (3, 5),
            (5, 3),
            (10, 5),
            (5, 10),
        ],
    )
    def test_various_matrix_shapes(self, m, n):
        """Test initialization with various matrix shapes."""
        X = Variable((m, n), name="X")
        X.value = torch.randn(m, n)
        nuc_norm = NucNorm(X, scaling=1.0)

        assert nuc_norm is not None


class TestNucNormForward:
    """Tests for forward evaluation."""

    def test_forward_diagonal_matrix(self, square_matrix):
        """Test forward with diagonal matrix."""
        nuc_norm, X = square_matrix

        result = nuc_norm.forward()

        # Diagonal matrix has singular values [1, 2, 3]
        # Nuclear norm = sum = 6.0
        expected = torch.tensor(6.0)
        assert torch.allclose(result, expected)

    def test_forward_identity_matrix(self):
        """Test forward with identity matrix."""
        X = Variable((4, 4), name="X")
        X.value = torch.eye(4)
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # Identity has all singular values = 1
        expected = torch.tensor(4.0)
        assert torch.allclose(result, expected)

    def test_forward_zero_matrix(self):
        """Test forward with zero matrix."""
        X = Variable((3, 3), name="X")
        X.value = torch.zeros(3, 3)
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # Zero matrix has all singular values = 0
        expected = torch.tensor(0.0)
        assert torch.allclose(result, expected)

    def test_forward_with_scaling(self):
        """Test forward with custom scaling factor."""
        X = Variable((3, 3), name="X")
        X.value = torch.eye(3)
        nuc_norm = NucNorm(X, scaling=2.0)

        result = nuc_norm.forward()

        # Identity has nuclear norm 3, scaled by 2
        expected = torch.tensor(6.0)
        assert torch.allclose(result, expected)

    def test_forward_rank_one_matrix(self):
        """Test forward with rank-1 matrix."""
        X = Variable((3, 3), name="X")
        # Rank-1 matrix: outer product
        u = torch.tensor([[1.0], [2.0], [3.0]])
        v = torch.tensor([[1.0, 1.0, 1.0]])
        X.value = u @ v
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # Rank-1 matrix has only one non-zero singular value
        # which equals the Frobenius norm
        expected = torch.linalg.norm(X.value, ord="fro")
        assert torch.allclose(result, expected, atol=1e-6)

    def test_forward_tall_matrix(self, tall_matrix):
        """Test forward with tall rectangular matrix."""
        nuc_norm, X = tall_matrix

        result = nuc_norm.forward()

        # Should compute sum of singular values
        S = torch.linalg.svdvals(X.value)
        expected = torch.sum(S)
        assert torch.allclose(result, expected)

    def test_forward_wide_matrix(self, wide_matrix):
        """Test forward with wide rectangular matrix."""
        nuc_norm, X = wide_matrix

        result = nuc_norm.forward()

        # Should compute sum of singular values
        S = torch.linalg.svdvals(X.value)
        expected = torch.sum(S)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("seed", range(5))
    def test_forward_random_matrix(self, seed):
        """Test forward with random matrices."""
        torch.manual_seed(seed)

        X = Variable((4, 6), name="X")
        X.value = torch.randn(4, 6)
        nuc_norm = NucNorm(X, scaling=0.5)

        result = nuc_norm.forward()

        # Manually compute expected
        S = torch.linalg.svdvals(X.value)
        expected = 0.5 * torch.sum(S)

        assert torch.allclose(result, expected)

    def test_forward_non_negative(self):
        """Test that forward always returns non-negative values."""
        X = Variable((3, 3), name="X")
        X.value = torch.randn(3, 3)
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # Nuclear norm is always non-negative
        assert result >= 0

    def test_forward_large_matrix(self):
        """Test forward with large matrix."""
        X = Variable((50, 30), name="X")
        X.value = torch.randn(50, 30)
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        S = torch.linalg.svdvals(X.value)
        expected = torch.sum(S)
        assert torch.allclose(result, expected)


class TestNucNormProperties:
    """Tests for nuclear norm properties."""

    def test_is_smooth_always_false(self, square_matrix):
        """Test that is_smooth always returns False."""
        nuc_norm, _ = square_matrix

        assert nuc_norm.is_smooth() is False

    def test_is_proxable_always_true(self, square_matrix):
        """Test that is_proxable always returns True."""
        nuc_norm, _ = square_matrix

        assert nuc_norm.is_proxable() is True

    def test_is_subsamplable_always_false(self, square_matrix):
        """Test that is_subsamplable always returns False."""
        nuc_norm, _ = square_matrix

        assert nuc_norm.is_subsamplable() is False

    @pytest.mark.parametrize("scaling", [0.1, 1.0, 10.0])
    def test_properties_consistent(self, scaling):
        """Test that properties are consistent across different scalings."""
        X = Variable((4, 4), name="X")
        X.value = torch.eye(4)
        nuc_norm = NucNorm(X, scaling=scaling)

        assert nuc_norm.is_smooth() is False
        assert nuc_norm.is_proxable() is True
        assert nuc_norm.is_subsamplable() is False


class TestNucNormProx:
    """Tests for proximal operator."""

    def test_prox_diagonal_matrix(self):
        """Test prox with diagonal matrix."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
        prox_scaling = 1.0
        projected = nuc_norm.prox(location, prox_scaling)

        # Soft-threshold singular values: [5-1, 3-1, 1-1] = [4, 2, 0]
        expected = torch.diag(torch.tensor([4.0, 2.0, 0.0]))
        assert torch.allclose(projected, expected, atol=1e-6)

    def test_prox_identity_matrix(self):
        """Test prox with identity matrix."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.eye(3)
        prox_scaling = 0.5
        projected = nuc_norm.prox(location, prox_scaling)

        # Soft-threshold all singular values (all = 1): [1-0.5, 1-0.5, 1-0.5]
        expected = 0.5 * torch.eye(3)
        assert torch.allclose(projected, expected, atol=1e-6)

    def test_prox_zero_matrix(self):
        """Test prox with zero matrix."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.zeros(3, 3)
        projected = nuc_norm.prox(location, prox_scaling=1.0)

        # Zero matrix should remain zero
        expected = torch.zeros(3, 3)
        assert torch.allclose(projected, expected)

    def test_prox_rank_reduction(self):
        """Test that prox reduces rank by thresholding small singular values."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        # Create matrix with known singular values
        U = torch.eye(3)
        S = torch.tensor([5.0, 2.0, 0.5])
        V = torch.eye(3)
        location = U @ torch.diag(S) @ V

        prox_scaling = 1.0
        projected = nuc_norm.prox(location, prox_scaling)

        # After soft-thresholding: [5-1, 2-1, 0.5-1] = [4, 1, 0]
        # Should reduce to rank 2
        rank = torch.linalg.matrix_rank(projected).item()
        assert rank == 2

    def test_prox_with_scaling(self):
        """Test prox with custom scaling factor."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=2.0)

        location = torch.diag(torch.tensor([10.0, 6.0, 2.0]))
        prox_scaling = 1.0
        projected = nuc_norm.prox(location, prox_scaling)

        # threshold = prox_scaling * scaling = 1.0 * 2.0 = 2.0
        # Soft-threshold: [10-2, 6-2, 2-2] = [8, 4, 0]
        expected = torch.diag(torch.tensor([8.0, 4.0, 0.0]))
        assert torch.allclose(projected, expected, atol=1e-6)

    @pytest.mark.parametrize("prox_scaling", [0.1, 0.5, 1.0, 2.0])
    def test_prox_various_scaling(self, prox_scaling):
        """Test prox with various prox_scaling values."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.diag(torch.tensor([5.0, 3.0, 1.0]))
        projected = nuc_norm.prox(location, prox_scaling)

        # Verify singular value soft-thresholding
        U, S, Vt = torch.linalg.svd(location, full_matrices=False)
        S_thresh = torch.nn.functional.relu(S - prox_scaling * 1.0)
        expected = (U * S_thresh) @ Vt

        assert torch.allclose(projected, expected, atol=1e-6)

    def test_prox_tall_matrix(self):
        """Test prox with tall rectangular matrix."""
        X = Variable((5, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.randn(5, 3)
        projected = nuc_norm.prox(location, prox_scaling=1.0)

        # Check shape is preserved
        assert projected.shape == location.shape

        # Verify soft-thresholding
        U, S, Vt = torch.linalg.svd(location, full_matrices=False)
        S_thresh = torch.nn.functional.relu(S - 1.0)
        expected = (U * S_thresh) @ Vt

        assert torch.allclose(projected, expected, atol=1e-6)

    def test_prox_wide_matrix(self):
        """Test prox with wide rectangular matrix."""
        X = Variable((3, 5), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.randn(3, 5)
        projected = nuc_norm.prox(location, prox_scaling=1.0)

        # Check shape is preserved
        assert projected.shape == location.shape

        # Verify soft-thresholding
        U, S, Vt = torch.linalg.svd(location, full_matrices=False)
        S_thresh = torch.nn.functional.relu(S - 1.0)
        expected = (U * S_thresh) @ Vt

        assert torch.allclose(projected, expected, atol=1e-6)

    def test_prox_shrinkage_property(self):
        """Test that prox reduces nuclear norm."""
        X = Variable((4, 4), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.randn(4, 4)
        projected = nuc_norm.prox(location, prox_scaling=1.0)

        # Nuclear norm should decrease
        norm_before = torch.sum(torch.linalg.svdvals(location))
        norm_after = torch.sum(torch.linalg.svdvals(projected))

        assert norm_after <= norm_before

    def test_prox_zero_prox_scaling(self):
        """Test prox with zero prox_scaling (identity operation)."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.randn(3, 3)
        projected = nuc_norm.prox(location, prox_scaling=0.0)

        # Should return location unchanged
        assert torch.allclose(projected, location)

    def test_prox_large_threshold(self):
        """Test prox with large threshold (should zero out matrix)."""
        X = Variable((3, 3), name="X")
        nuc_norm = NucNorm(X, scaling=1.0)

        location = torch.randn(3, 3)
        # Use very large prox_scaling to threshold all singular values
        max_singular_value = torch.linalg.svdvals(location).max()
        projected = nuc_norm.prox(location, prox_scaling=max_singular_value + 1.0)

        # Should be close to zero
        assert torch.allclose(projected, torch.zeros(3, 3), atol=1e-6)


class TestNucNormSubsample:
    """Tests for subsampling (should raise error)."""

    def test_subsample_raises_not_implemented(self, square_matrix):
        """Test that subsample raises NotImplementedError."""
        nuc_norm, _ = square_matrix

        with pytest.raises(NotImplementedError, match="cannot be subsampled"):
            nuc_norm.subsample(torch.tensor([0, 1]))


class TestNucNormEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, dtype):
        """Test with different floating point precisions."""
        X = Variable((3, 3), name="X")
        X.value = torch.eye(3, dtype=dtype)
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()
        assert result.dtype == dtype

        location = torch.eye(3, dtype=dtype)
        projected = nuc_norm.prox(location, prox_scaling=0.5)
        assert projected.dtype == dtype

    def test_very_small_singular_values(self):
        """Test with matrix having very small singular values."""
        X = Variable((3, 3), name="X")
        # Create matrix with small singular values
        U = torch.eye(3)
        S = torch.tensor([1e-8, 1e-9, 1e-10])
        V = torch.eye(3)
        X.value = U @ torch.diag(S) @ V
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # Should handle small values correctly
        expected = torch.sum(S)
        assert torch.allclose(result, expected, atol=1e-12)

    def test_very_large_singular_values(self):
        """Test with matrix having very large singular values."""
        X = Variable((3, 3), name="X")
        X.value = torch.diag(torch.tensor([1e6, 1e5, 1e4]))
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        expected = torch.tensor(1.11e6)
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_near_singular_matrix(self):
        """Test with near-singular matrix (very small smallest singular value)."""
        X = Variable((3, 3), name="X")
        # Create near-singular matrix
        X.value = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0 + 1e-6]]
        )
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # Should handle near-singular case
        S = torch.linalg.svdvals(X.value)
        expected = torch.sum(S)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_symmetric_matrix(self):
        """Test with symmetric matrix."""
        X = Variable((3, 3), name="X")
        X.value = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        # For symmetric matrix, singular values = |eigenvalues|
        S = torch.linalg.svdvals(X.value)
        expected = torch.sum(S)
        assert torch.allclose(result, expected)

    def test_antisymmetric_matrix(self):
        """Test with antisymmetric (skew-symmetric) matrix."""
        X = Variable((3, 3), name="X")
        X.value = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]])
        nuc_norm = NucNorm(X, scaling=1.0)

        result = nuc_norm.forward()

        S = torch.linalg.svdvals(X.value)
        expected = torch.sum(S)
        assert torch.allclose(result, expected)


class TestNucNormExamples:
    """Tests based on docstring examples."""

    def test_docstring_basic_example(self):
        """Test the basic example from docstring."""
        X = Variable((10, 5), name="X")
        X.value = torch.randn(10, 5)
        nuc_norm = NucNorm(X, scaling=0.1)

        loss = nuc_norm.forward()

        assert loss >= 0
        assert torch.isfinite(loss)

        # Verify scaling is applied
        S = torch.linalg.svdvals(X.value)
        expected = 0.1 * torch.sum(S)
        assert torch.allclose(loss, expected)


@pytest.mark.parametrize(
    "m,n,scaling,seed",
    [
        (3, 3, 1.0, 0),
        (5, 3, 0.5, 1),
        (3, 5, 0.5, 2),
        (10, 5, 0.1, 3),
        (5, 10, 0.1, 4),
    ],
)
def test_nucnorm_general(m, n, scaling, seed):
    """General test for NucNorm with various configurations."""
    torch.manual_seed(seed)

    X = Variable((m, n), name="X")
    X.value = torch.randn(m, n)
    nuc_norm = NucNorm(X, scaling=scaling)

    # Test forward
    result = nuc_norm.forward()
    S = torch.linalg.svdvals(X.value)
    expected = scaling * torch.sum(S)
    assert torch.allclose(result, expected)

    # Test properties
    assert nuc_norm.is_smooth() is False
    assert nuc_norm.is_proxable() is True
    assert nuc_norm.is_subsamplable() is False

    # Test prox
    location = torch.randn(m, n)
    projected = nuc_norm.prox(location, prox_scaling=1.0)

    # Verify shape preservation
    assert projected.shape == location.shape

    # Verify singular value soft-thresholding
    U, S_loc, Vt = torch.linalg.svd(location, full_matrices=False)
    S_thresh = torch.nn.functional.relu(S_loc - 1.0 * scaling)
    expected_prox = (U * S_thresh) @ Vt
    assert torch.allclose(projected, expected_prox, atol=1e-6)

    # Verify nuclear norm reduction
    norm_before = torch.sum(torch.linalg.svdvals(location))
    norm_after = torch.sum(torch.linalg.svdvals(projected))
    assert norm_after <= norm_before + 1e-6
