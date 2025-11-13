"""Tests for NucNorm atom."""

import pytest
import torch

from rlaopt.atoms.nuc_norm import NucNorm
from rlaopt.expression import ExprTree, Variable


@pytest.fixture
def matrix_var():
    """Create a matrix variable."""
    return Variable((4, 3), name="X")


@pytest.fixture
def square_matrix_var():
    """Create a square matrix variable."""
    return Variable((5, 5), name="A")


@pytest.fixture
def wide_matrix_var():
    """Create a wide matrix variable (more columns than rows)."""
    return Variable((3, 5), name="W")


class TestNucNorm:
    """Test NucNorm atom."""

    # ----------------------
    # Initialization tests
    # ----------------------

    def test_init_with_matrix_variable(self, matrix_var):
        """Test initialization with 2D matrix variable."""
        nuc = NucNorm(matrix_var)
        assert nuc.scaling == 1.0

    def test_init_with_custom_scaling(self, matrix_var):
        """Test initialization with custom scaling factor."""
        nuc = NucNorm(matrix_var, scaling=0.5)
        assert nuc.scaling == 0.5

    def test_init_with_tensor_scaling(self, matrix_var):
        """Test initialization with tensor scaling."""
        scaling = torch.tensor(2.0)
        nuc = NucNorm(matrix_var, scaling=scaling)
        assert torch.equal(nuc.scaling, scaling)

    def test_init_raises_error_for_non_2d_variable(self):
        """Test initialization raises ValueError for non-2D variables."""
        vector_var = Variable((10,), name="x")
        with pytest.raises(ValueError, match="must be 2D Tensor"):
            NucNorm(vector_var)

    # ----------------------
    # Forward evaluation tests
    # ----------------------

    def test_forward_computes_sum_of_singular_values(self, matrix_var):
        """Test forward() computes sum of singular values."""
        matrix_var.value.data = torch.eye(
            4, 3
        )  # Identity-like, singular values = [1, 1, 1]
        nuc = NucNorm(matrix_var)
        result = nuc.forward()
        assert torch.allclose(result, torch.tensor(3.0))

    def test_forward_with_scaling(self, matrix_var):
        """Test forward() applies scaling factor."""
        matrix_var.value.data = torch.eye(4, 3)
        nuc = NucNorm(matrix_var, scaling=2.0)
        result = nuc.forward()
        assert torch.allclose(result, torch.tensor(6.0))

    def test_forward_with_rank_deficient_matrix(self, matrix_var):
        """Test forward() with rank-deficient matrix."""
        # Create rank-1 matrix
        matrix_var.value.data = torch.ones(4, 3)
        nuc = NucNorm(matrix_var)
        result = nuc.forward()
        # For ones matrix 4x3, largest singular value ≈ sqrt(12) ≈ 3.46
        assert result > 0

    def test_forward_with_zero_matrix(self, matrix_var):
        """Test forward() with zero matrix."""
        matrix_var.value.data = torch.zeros(4, 3)
        nuc = NucNorm(matrix_var)
        result = nuc.forward()
        assert torch.allclose(result, torch.tensor(0.0))

    # ----------------------
    # Property tests
    # ----------------------

    def test_is_smooth_returns_false(self, matrix_var):
        """Test nuclear norm is not smooth."""
        nuc = NucNorm(matrix_var)
        assert nuc.is_smooth() is False

    def test_is_proxable_returns_true(self, matrix_var):
        """Test nuclear norm is proxable."""
        nuc = NucNorm(matrix_var)
        assert nuc.is_proxable() is True

    def test_is_subsamplable_returns_false(self, matrix_var):
        """Test nuclear norm is not subsamplable."""
        nuc = NucNorm(matrix_var)
        assert nuc.is_subsamplable() is False

    def test_subsample_raises_not_implemented(self, matrix_var):
        """Test subsample() raises NotImplementedError."""
        nuc = NucNorm(matrix_var)
        with pytest.raises(NotImplementedError, match="cannot be subsampled"):
            nuc.subsample(torch.tensor([0, 1]))

    # ----------------------
    # Proximal operator tests
    # ----------------------

    def test_prox_performs_soft_thresholding(self, matrix_var):
        """Test prox() performs singular value soft-thresholding."""
        nuc = NucNorm(matrix_var, scaling=1.0)
        location = torch.diag(torch.tensor([5.0, 3.0, 2.0]))  # Diagonal matrix
        location = torch.cat([location, torch.zeros(1, 3)], dim=0)  # Make it 4x3

        result = nuc.prox(location, prox_scaling=1.0)
        # Singular values [5, 3, 2] thresholded by 1.0 → [4, 2, 1]
        S = torch.linalg.svdvals(result)
        expected_S = torch.tensor([4.0, 2.0, 1.0])
        assert torch.allclose(S, expected_S, atol=1e-5)

    def test_prox_with_scaling_factor(self, matrix_var):
        """Test prox() with different scaling factors."""
        nuc = NucNorm(matrix_var, scaling=2.0)
        location = torch.diag(torch.tensor([10.0, 8.0, 6.0]))
        location = torch.cat([location, torch.zeros(1, 3)], dim=0)

        result = nuc.prox(location, prox_scaling=1.0)
        # Threshold = 2.0 * 1.0 = 2.0
        # Singular values [10, 8, 6] thresholded by 2.0 → [8, 6, 4]
        S = torch.linalg.svdvals(result)
        expected_S = torch.tensor([8.0, 6.0, 4.0])
        assert torch.allclose(S, expected_S, atol=1e-5)

    def test_prox_handles_wide_matrices(self, wide_matrix_var):
        """Test prox() handles wide matrices (m < n)."""
        nuc = NucNorm(wide_matrix_var, scaling=1.0)
        location = torch.randn(3, 5)
        result = nuc.prox(location, prox_scaling=0.5)

        # Check result has correct shape
        assert result.shape == (3, 5)

        # Check singular values are thresholded
        S_in = torch.linalg.svdvals(location)
        S_out = torch.linalg.svdvals(result)
        assert torch.all(S_out <= S_in)

    def test_prox_zeros_small_singular_values(self, matrix_var):
        """Test prox() zeros out singular values below threshold."""
        nuc = NucNorm(matrix_var, scaling=1.0)
        location = torch.diag(torch.tensor([3.0, 0.5, 0.2]))
        location = torch.cat([location, torch.zeros(1, 3)], dim=0)

        result = nuc.prox(location, prox_scaling=1.0)
        S = torch.linalg.svdvals(result)
        # Values [3.0, 0.5, 0.2] thresholded by 1.0 → [2.0, 0, 0]
        expected_S = torch.tensor([2.0, 0.0, 0.0])
        assert torch.allclose(S, expected_S, atol=1e-5)

    def test_prox_gradient_flows_through(self, matrix_var):
        """Test prox() allows gradient computation."""
        nuc = NucNorm(matrix_var, scaling=1.0)
        location = torch.randn(4, 3, requires_grad=True)

        result = nuc.prox(location, prox_scaling=0.5)
        loss = result.sum()
        loss.backward()

        assert location.grad is not None
        assert not torch.isnan(location.grad).any()

    # ----------------------
    # Edge cases
    # ----------------------

    def test_with_square_matrix(self, square_matrix_var):
        """Test nuclear norm works with square matrices."""
        square_matrix_var.value.data = torch.eye(5)
        nuc = NucNorm(square_matrix_var)
        result = nuc.forward()
        assert torch.allclose(result, torch.tensor(5.0))

    def test_prox_preserves_zero_matrix(self, matrix_var):
        """Test prox() of zero matrix is zero."""
        nuc = NucNorm(matrix_var, scaling=1.0)
        location = torch.zeros(4, 3)
        result = nuc.prox(location, prox_scaling=1.0)
        assert torch.allclose(result, torch.zeros(4, 3))


class TestNucNormScaling:
    """Tests for NucNorm scalar multiplication."""

    def test_scalar_multiplication_optimization(self, matrix_var):
        """Test that scalar multiplication optimizes to scaled NucNorm."""
        # Set to identity-like matrix with known singular values
        matrix_var.value.data = torch.eye(4, 3)
        nuc = NucNorm(matrix_var, scaling=2.0)
        scaled = 4.0 * nuc

        assert isinstance(scaled, NucNorm)
        assert scaled.tree() == ExprTree("NucNorm", ExprTree("Variable(X)"))
        assert torch.allclose(scaled.forward(), torch.tensor(24.0))
