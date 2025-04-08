import pytest
import torch
from rlaopt.preconditioners.skpre import SkPre
from rlaopt.preconditioners.configs import SkPreConfig


def get_available_devices():
    """Return a list of available devices to test."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    return devices


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parameterized fixture for testing on different devices."""
    return torch.device(request.param)


@pytest.fixture
def test_matrix(device):
    """Create a test matrix."""
    return torch.randn(100, 50, device=device)


@pytest.fixture
def test_vector(device):
    """Create a test vector matching the columns of test_matrix."""
    return torch.randn(50, device=device)


@pytest.fixture
def test_matrix_batch(device):
    """Create a batch of test vectors matching the columns of test_matrix."""
    return torch.randn(50, 5, device=device)


@pytest.fixture(params=["gauss", "ortho", "sparse"])
def sketch_type(request):
    """Parameterized fixture for different sketch types."""
    return request.param


@pytest.fixture
def small_sketch_config(sketch_type):
    """Create a configuration with sketch_size < dimension."""
    return SkPreConfig(sketch=sketch_type, sketch_size=20, rho=1e-4)


@pytest.fixture
def large_sketch_config(sketch_type):
    """Create a configuration with sketch_size > dimension."""
    return SkPreConfig(sketch=sketch_type, sketch_size=100, rho=1e-4)


class TestSkPreBasics:
    """Basic tests for SkPre."""

    def test_initialization(self, small_sketch_config):
        """Test that SkPre initializes correctly."""
        precond = SkPre(small_sketch_config)
        assert precond.Y is None
        assert precond.L is None

    def test_update_small_sketch(self, test_matrix, small_sketch_config, device):
        """Test updating with small sketch size."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        # With small sketch size, Y should exist
        assert precond.Y is not None
        assert precond.L is not None
        assert precond.Y.device == device
        assert precond.L.device == device
        # Check dimensions
        assert precond.Y.shape[0] == small_sketch_config.sketch_size
        assert precond.Y.shape[1] == test_matrix.shape[1]
        assert precond.L.shape == (
            small_sketch_config.sketch_size,
            small_sketch_config.sketch_size,
        )

    def test_update_large_sketch(self, test_matrix, large_sketch_config, device):
        """Test updating with large sketch size."""
        precond = SkPre(large_sketch_config)
        precond._update(test_matrix, device)

        # With large sketch size, Y should be deleted
        assert precond.Y is None
        assert precond.L is not None
        assert precond.L.device == device
        # Check dimensions
        assert precond.L.shape == (test_matrix.shape[1], test_matrix.shape[1])


class TestSkPreOperations:
    """Tests for SkPre matrix operations."""

    def test_matmul_vector(self, test_matrix, test_vector, small_sketch_config, device):
        """Test matrix multiplication with a vector."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond @ test_vector

        # Check shape and device
        assert result.shape == test_vector.shape
        assert result.device == device

    def test_matmul_batch(
        self, test_matrix, test_matrix_batch, small_sketch_config, device
    ):
        """Test matrix multiplication with a batch of vectors."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond @ test_matrix_batch

        # Check shape and device
        assert result.shape == test_matrix_batch.shape
        assert result.device == device

    def test_inverse_matmul_vector(
        self, test_matrix, test_vector, small_sketch_config, device
    ):
        """Test inverse matrix multiplication with a vector."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_vector

        # Check shape and device
        assert result.shape == test_vector.shape
        assert result.device == device

    def test_inverse_matmul_batch(
        self, test_matrix, test_matrix_batch, small_sketch_config, device
    ):
        """Test inverse matrix multiplication with a batch of vectors."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_matrix_batch

        # Check shape and device
        assert result.shape == test_matrix_batch.shape
        assert result.device == device


class TestSkPreConsistency:
    """Tests for mathematical consistency of SkPre."""

    def test_inverse_consistency_small_sketch(
        self, test_matrix, test_vector, small_sketch_config, device
    ):
        """Test P @ (P^-1 @ x) ≈ x with small sketch size."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        # Apply inverse then apply preconditioner
        inverse_x = precond._inv @ test_vector
        reconstructed_x = precond @ inverse_x

        # Check if we get back approximately the original vector
        assert torch.allclose(reconstructed_x, test_vector, rtol=1e-3, atol=1e-3)

    def test_inverse_consistency_large_sketch(
        self, test_matrix, test_vector, large_sketch_config, device
    ):
        """Test P @ (P^-1 @ x) ≈ x with large sketch size."""
        precond = SkPre(large_sketch_config)
        precond._update(test_matrix, device)

        # Apply inverse then apply preconditioner
        inverse_x = precond._inv @ test_vector
        reconstructed_x = precond @ inverse_x

        # Check if we get back approximately the original vector
        assert torch.allclose(reconstructed_x, test_vector, rtol=1e-3, atol=1e-3)


class TestSkPreMemory:
    """Simple test for memory management in SkPre."""

    def test_del_Y(self, test_matrix, small_sketch_config, device):
        """Test that _del_Y properly removes Y."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        # Verify Y exists
        assert precond.Y is not None

        # Call _del_Y
        precond._del_Y()

        # Verify Y is gone
        assert precond.Y is None
