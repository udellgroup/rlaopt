import pytest
import torch
from rlaopt.preconditioners.nystrom import Nystrom
from rlaopt.preconditioners.configs import NystromConfig
from rlaopt.linops import SymmetricLinOp


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
    """Create a positive semidefinite test matrix."""
    A = torch.randn(50, 50, device=device, dtype=precision)
    return A @ A.T  # Make it positive semidefinite


@pytest.fixture
def test_vector(device, precision):
    """Create a test vector matching the dimensions of test_matrix."""
    return torch.randn(50, device=device, dtype=precision)


@pytest.fixture
def test_matrix_batch(device, precision):
    """Create a batch of test vectors."""
    return torch.randn(50, 5, device=device, dtype=precision)


@pytest.fixture(params=["gauss", "ortho", "sparse"])
def sketch_type(request):
    """Parameterized fixture for different sketch types."""
    return request.param


@pytest.fixture
def nystrom_config(sketch_type):
    """Create a Nystrom preconditioner configuration."""
    return NystromConfig(
        rank=20, sketch=sketch_type, rho=1e-2, damping_mode="non_adaptive"
    )


@pytest.fixture
def adaptive_nystrom_config(sketch_type):
    """Create a Nystrom preconditioner configuration with adaptive damping."""
    return NystromConfig(rank=20, sketch=sketch_type, rho=1e-2, damping_mode="adaptive")


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-6},
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


@pytest.fixture
def tol(precision):
    """Return appropriate tolerance values for the current precision."""
    return TOLERANCES[precision]


@pytest.fixture
def symmetric_linop(test_matrix):
    """Create a symmetric linear operator."""

    def matvec(x):
        if x.ndim == 1:
            return test_matrix @ x
        else:  # x.ndim == 2
            return test_matrix @ x

    shape = test_matrix.shape
    device = test_matrix.device

    return SymmetricLinOp(device, shape, matvec, dtype=test_matrix.dtype)


class TestNystromBasics:
    """Basic tests for Nystrom preconditioner."""

    def test_initialization(self, nystrom_config):
        """Test that the Nystrom preconditioner initializes correctly."""
        precond = Nystrom(nystrom_config)
        assert precond.U is None
        assert precond.S is None

    def test_update(self, test_matrix, nystrom_config, device, precision, tol):
        """Test updating the Nystrom preconditioner."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        # After update, U and S should exist
        assert precond.U is not None
        assert precond.S is not None
        assert precond.U.device == device
        assert precond.S.device == device
        assert precond.U.dtype == precision
        assert precond.S.dtype == precision

        # Check dimensions
        assert precond.U.shape == (test_matrix.shape[0], nystrom_config.rank)
        assert precond.S.shape == (nystrom_config.rank,)

        # S should be non-negative
        assert torch.all(precond.S >= 0)

        # U should have orthonormal columns
        UTU = precond.U.T @ precond.U
        identity = torch.eye(nystrom_config.rank, device=device, dtype=precision)
        assert torch.allclose(UTU, identity, **tol)


class TestNystromOperations:
    """Tests for Nystrom preconditioner matrix operations."""

    def test_matmul_vector(
        self, test_matrix, test_vector, nystrom_config, device, precision, tol
    ):
        """Test matrix multiplication with a vector."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        result = precond @ test_vector

        # Check shape, device and precision
        assert result.shape == test_vector.shape
        assert result.device == device
        assert result.dtype == precision

        # Verify it matches manual computation
        expected = (
            precond.U @ (precond.S * (precond.U.T @ test_vector))
            + nystrom_config.rho * test_vector
        )
        assert torch.allclose(result, expected, **tol)

    def test_matmul_batch(
        self, test_matrix, test_matrix_batch, nystrom_config, device, precision, tol
    ):
        """Test matrix multiplication with a batch of vectors."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        result = precond @ test_matrix_batch

        # Check shape, device and precision
        assert result.shape == test_matrix_batch.shape
        assert result.device == device
        assert result.dtype == precision

        # Verify it matches manual computation
        expected = (
            precond.U @ (precond.S[:, None] * (precond.U.T @ test_matrix_batch))
            + nystrom_config.rho * test_matrix_batch
        )
        assert torch.allclose(result, expected, **tol)

    def test_inverse_matmul_vector(
        self, test_matrix, test_vector, nystrom_config, device, precision
    ):
        """Test inverse matrix multiplication with a vector."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_vector

        # Check shape, device and precision
        assert result.shape == test_vector.shape
        assert result.device == device
        assert result.dtype == precision

    def test_inverse_matmul_batch(
        self, test_matrix, test_matrix_batch, nystrom_config, device, precision
    ):
        """Test inverse matrix multiplication with a batch of vectors."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_matrix_batch

        # Check shape, device and precision
        assert result.shape == test_matrix_batch.shape
        assert result.device == device
        assert result.dtype == precision


class TestNystromConsistency:
    """Tests for mathematical consistency of Nystrom preconditioner."""

    def test_inverse_consistency(
        self, test_matrix, test_vector, nystrom_config, device, precision, tol
    ):
        """Test P @ (P^-1 @ x) ≈ x."""
        precond = Nystrom(nystrom_config)
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

    def test_low_rank_approximation(
        self, test_matrix, nystrom_config, device, precision, tol
    ):
        """Test that the preconditioner provides a low-rank approximation."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        # Create the explicit low-rank component of the preconditioner
        n = test_matrix.shape[0]
        low_rank_matrix = precond.U @ torch.diag(precond.S) @ precond.U.T

        # Add the diagonal rho term
        explicit_precond = low_rank_matrix + nystrom_config.rho * torch.eye(
            n, device=device, dtype=precision
        )

        # Test the effect on random vectors to avoid creating the full matrix
        for _ in range(5):
            x = torch.randn(n, device=device, dtype=precision)

            # Apply the explicit preconditioner
            explicit_result = explicit_precond @ x

            # Apply the actual preconditioner
            actual_result = precond @ x

            # Results should match
            assert torch.allclose(explicit_result, actual_result, **tol)


class TestNystromDamping:
    """Tests for damping behavior in Nystrom preconditioner."""

    def test_fixed_damping(self, test_matrix, nystrom_config, device):
        """Test fixed damping mode."""
        precond = Nystrom(nystrom_config)
        precond._update(test_matrix, device)

        # Store initial rho
        initial_rho = precond.config.rho

        # Update damping
        precond._update_damping(2.0)

        # With fixed damping, rho should remain unchanged
        assert precond.config.rho == initial_rho

    def test_adaptive_damping(
        self, test_matrix, adaptive_nystrom_config, device, precision
    ):
        """Test adaptive damping mode."""
        precond = Nystrom(adaptive_nystrom_config)
        precond._update(test_matrix, device)

        # Store the smallest eigenvalue
        smallest_eigval = precond.S[-1]

        # Update damping
        baseline_rho = 2.0
        precond._update_damping(baseline_rho)

        # With adaptive damping, rho should be baseline_rho + smallest_eigval
        assert torch.isclose(
            torch.tensor(precond.config.rho, device=device, dtype=precision),
            torch.tensor(
                baseline_rho + smallest_eigval, device=device, dtype=precision
            ),
        )


class TestNystromWithLinOp:
    """Tests for Nystrom preconditioner with linear operators."""

    def test_update_with_linop(
        self, symmetric_linop, nystrom_config, device, precision, tol
    ):
        """Test updating the Nystrom preconditioner with a symmetric linear operator."""
        # Initialize and update the preconditioner with the linear operator
        precond = Nystrom(nystrom_config)
        precond._update(symmetric_linop, device)

        # Verify basic properties
        assert precond.U is not None
        assert precond.S is not None
        assert precond.U.device == device
        assert precond.S.device == device
        assert precond.U.dtype == precision
        assert precond.S.dtype == precision
        assert precond.U.shape == (symmetric_linop.shape[0], nystrom_config.rank)
        assert precond.S.shape == (nystrom_config.rank,)

        # Verify orthonormality of U
        UTU = precond.U.T @ precond.U
        identity = torch.eye(nystrom_config.rank, device=device, dtype=precision)
        assert torch.allclose(UTU, identity, **tol)

        # Test inverse consistency with random vectors
        for _ in range(3):  # Test with a few different vectors
            x = torch.randn(symmetric_linop.shape[0], device=device, dtype=precision)

            # Apply inverse then apply preconditioner
            inverse_x = precond._inv @ x
            reconstructed_x = precond @ inverse_x

            # Check if we get back approximately the original vector
            assert torch.allclose(reconstructed_x, x, **tol)

        # Test batch inverse consistency
        X = torch.randn(symmetric_linop.shape[0], 5, device=device, dtype=precision)
        inverse_X = precond._inv @ X
        reconstructed_X = precond @ inverse_X
        assert torch.allclose(reconstructed_X, X, **tol)
