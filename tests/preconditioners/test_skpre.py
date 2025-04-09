import pytest
import torch

from rlaopt.preconditioners import SkPreConfig
from rlaopt.preconditioners.skpre import SkPre


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


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    """Parameterized fixture for testing with different precision."""
    return request.param


@pytest.fixture
def test_matrix(device, precision):
    """Create a test matrix with the specified precision."""
    return torch.randn(100, 50, device=device, dtype=precision) / (100**0.5)


@pytest.fixture
def test_vector(device, precision):
    """Create a test vector with the specified precision."""
    return torch.randn(50, device=device, dtype=precision)


@pytest.fixture
def test_matrix_batch(device, precision):
    """Create a batch of test vectors with the specified precision."""
    return torch.randn(50, 5, device=device, dtype=precision)


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


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {
        "rtol": 1e-2,
        "atol": 1e-4,
    },  # tolerances are large for float32: sketch-and-precondition
    # can be unstable in single precision
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


@pytest.fixture
def tol(precision):
    """Return appropriate tolerance values for the current precision."""
    return TOLERANCES[precision]


class TestSkPreBasics:
    """Basic tests for SkPre."""

    def test_initialization(self, small_sketch_config):
        """Test that SkPre initializes correctly."""
        precond = SkPre(small_sketch_config)
        assert precond.Y is None
        assert precond.L is None

    def test_update_small_sketch(
        self, test_matrix, small_sketch_config, device, precision
    ):
        """Test updating with small sketch size."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        assert precond.Y is None
        assert precond.L is not None
        assert precond.L.device == device
        assert precond.L.dtype == precision
        # Check dimensions
        assert precond.L.shape == (test_matrix.shape[1], test_matrix.shape[1])

    def test_update_large_sketch(
        self, test_matrix, large_sketch_config, device, precision
    ):
        """Test updating with large sketch size."""
        precond = SkPre(large_sketch_config)
        precond._update(test_matrix, device)

        assert precond.Y is None
        assert precond.L is not None
        assert precond.L.device == device
        assert precond.L.dtype == precision
        # Check dimensions
        assert precond.L.shape == (test_matrix.shape[1], test_matrix.shape[1])


class TestSkPreOperations:
    """Tests for SkPre matrix operations."""

    def test_matmul_vector(
        self, test_matrix, test_vector, small_sketch_config, device, precision
    ):
        """Test matrix multiplication with a vector."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond @ test_vector

        # Check shape, device and precision
        assert result.shape == test_vector.shape
        assert result.device == device
        assert result.dtype == precision

    def test_matmul_batch(
        self, test_matrix, test_matrix_batch, small_sketch_config, device, precision
    ):
        """Test matrix multiplication with a batch of vectors."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond @ test_matrix_batch

        # Check shape, device and precision
        assert result.shape == test_matrix_batch.shape
        assert result.device == device
        assert result.dtype == precision

    def test_inverse_matmul_vector(
        self, test_matrix, test_vector, small_sketch_config, device, precision
    ):
        """Test inverse matrix multiplication with a vector."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_vector

        # Check shape, device and precision
        assert result.shape == test_vector.shape
        assert result.device == device
        assert result.dtype == precision

    def test_inverse_matmul_batch(
        self, test_matrix, test_matrix_batch, small_sketch_config, device, precision
    ):
        """Test inverse matrix multiplication with a batch of vectors."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_matrix_batch

        # Check shape, device and precision
        assert result.shape == test_matrix_batch.shape
        assert result.device == device
        assert result.dtype == precision


class TestSkPreConsistency:
    """Tests for mathematical consistency of SkPre."""

    def test_inverse_consistency_small_sketch(
        self, test_matrix, test_vector, small_sketch_config, device, precision, tol
    ):
        """Test P @ (P^-1 @ x) ≈ x with small sketch size."""
        precond = SkPre(small_sketch_config)
        precond._update(test_matrix, device)

        # Apply inverse then apply preconditioner
        inverse_x = precond._inv @ test_vector

        # Verify precision is preserved
        assert inverse_x.dtype == precision

        reconstructed_x = precond @ inverse_x

        # Verify precision is preserved
        assert reconstructed_x.dtype == precision

        # Check if we get back approximately the original vector
        assert torch.allclose(reconstructed_x, test_vector, **tol)

    def test_inverse_consistency_large_sketch(
        self, test_matrix, test_vector, large_sketch_config, device, precision, tol
    ):
        """Test P @ (P^-1 @ x) ≈ x with large sketch size."""
        precond = SkPre(large_sketch_config)
        precond._update(test_matrix, device)

        # Apply inverse then apply preconditioner
        inverse_x = precond._inv @ test_vector

        # Verify precision is preserved
        assert inverse_x.dtype == precision

        reconstructed_x = precond @ inverse_x

        # Verify precision is preserved
        assert reconstructed_x.dtype == precision

        # Check if we get back approximately the original vector
        assert torch.allclose(reconstructed_x, test_vector, **tol)
