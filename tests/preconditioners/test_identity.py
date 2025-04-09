import pytest
import torch

from rlaopt.preconditioners import IdentityConfig
from rlaopt.preconditioners.identity import Identity


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
def identity_config():
    """Create an Identity preconditioner configuration."""
    return IdentityConfig()


@pytest.fixture
def test_vector(device, precision):
    """Create a test vector."""
    return torch.randn(50, device=device, dtype=precision)


@pytest.fixture
def test_matrix_batch(device, precision):
    """Create a batch of test vectors."""
    return torch.randn(50, 5, device=device, dtype=precision)


class TestIdentityPreconditioner:
    """Tests for Identity preconditioner."""

    def test_update(self, identity_config, device):
        """Test update method (should be a no-op)."""
        precond = Identity(identity_config)
        # Just checking it doesn't raise errors
        precond._update(None, device)

    def test_matmul_vector(self, identity_config, test_vector):
        """Test matrix multiplication with a vector."""
        precond = Identity(identity_config)
        result = precond @ test_vector

        # For Identity, result should be identical to input
        assert torch.equal(result, test_vector)

        # Check device and precision are preserved
        assert result.device == test_vector.device
        assert result.dtype == test_vector.dtype

    def test_matmul_batch(self, identity_config, test_matrix_batch):
        """Test matrix multiplication with a batch of vectors."""
        precond = Identity(identity_config)
        result = precond @ test_matrix_batch

        # For Identity, result should be identical to input
        assert torch.equal(result, test_matrix_batch)

        # Check device and precision are preserved
        assert result.device == test_matrix_batch.device
        assert result.dtype == test_matrix_batch.dtype

    def test_inverse_matmul_vector(self, identity_config, test_vector):
        """Test inverse matrix multiplication with a vector."""
        precond = Identity(identity_config)
        result = precond._inv @ test_vector

        # For Identity, inverse result should be identical to input
        assert torch.equal(result, test_vector)

        # Check device and precision are preserved
        assert result.device == test_vector.device
        assert result.dtype == test_vector.dtype

    def test_inverse_matmul_batch(self, identity_config, test_matrix_batch):
        """Test inverse matrix multiplication with a batch of vectors."""
        precond = Identity(identity_config)
        result = precond._inv @ test_matrix_batch

        # For Identity, inverse result should be identical to input
        assert torch.equal(result, test_matrix_batch)

        # Check device and precision are preserved
        assert result.device == test_matrix_batch.device
        assert result.dtype == test_matrix_batch.dtype
