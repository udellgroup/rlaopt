import pytest
import torch

from rlaopt.preconditioners.newton import Newton
from rlaopt.preconditioners.configs import NewtonConfig
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
    """Create a positive definite test matrix."""
    A = torch.randn(50, 50, device=device, dtype=precision)
    # Make it positive definite
    return A @ A.T + torch.eye(50, device=device, dtype=precision)


@pytest.fixture
def test_vector(device, precision):
    """Create a test vector matching the dimensions of test_matrix."""
    return torch.randn(50, device=device, dtype=precision)


@pytest.fixture
def test_matrix_batch(device, precision):
    """Create a batch of test vectors."""
    return torch.randn(50, 5, device=device, dtype=precision)


@pytest.fixture
def newton_config():
    """Create a Newton preconditioner configuration."""
    return NewtonConfig(rho=1e-4)


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


class TestNewtonBasics:
    """Basic tests for Newton preconditioner."""

    def test_initialization(self, newton_config):
        """Test that the Newton preconditioner initializes correctly."""
        precond = Newton(newton_config)
        assert precond.L is None

    def test_update(self, test_matrix, newton_config, device, precision):
        """Test updating the Newton preconditioner."""
        precond = Newton(newton_config)
        precond._update(test_matrix, device)

        # After update, L should exist
        assert precond.L is not None
        assert precond.L.device == device
        assert precond.L.dtype == precision

        # L should be lower triangular
        assert torch.allclose(precond.L, torch.tril(precond.L))

        # Check dimensions
        assert precond.L.shape == test_matrix.shape


class TestNewtonOperations:
    """Tests for Newton preconditioner matrix operations."""

    def test_matmul_vector(
        self, test_matrix, test_vector, newton_config, device, precision, tol
    ):
        """Test matrix multiplication with a vector."""
        precond = Newton(newton_config)
        precond._update(test_matrix, device)

        result = precond @ test_vector

        # Check shape, device and precision
        assert result.shape == test_vector.shape
        assert result.device == device
        assert result.dtype == precision

        # Verify it matches manual computation
        expected = precond.L @ (precond.L.T @ test_vector)
        assert torch.allclose(result, expected, **tol)

    def test_matmul_batch(
        self, test_matrix, test_matrix_batch, newton_config, device, precision, tol
    ):
        """Test matrix multiplication with a batch of vectors."""
        precond = Newton(newton_config)
        precond._update(test_matrix, device)

        result = precond @ test_matrix_batch

        # Check shape, device and precision
        assert result.shape == test_matrix_batch.shape
        assert result.device == device
        assert result.dtype == precision

        # Verify it matches manual computation
        expected = precond.L @ (precond.L.T @ test_matrix_batch)
        assert torch.allclose(result, expected, **tol)

    def test_inverse_matmul_vector(
        self, test_matrix, test_vector, newton_config, device, precision
    ):
        """Test inverse matrix multiplication with a vector."""
        precond = Newton(newton_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_vector

        # Check shape, device and precision
        assert result.shape == test_vector.shape
        assert result.device == device
        assert result.dtype == precision

    def test_inverse_matmul_batch(
        self, test_matrix, test_matrix_batch, newton_config, device, precision
    ):
        """Test inverse matrix multiplication with a batch of vectors."""
        precond = Newton(newton_config)
        precond._update(test_matrix, device)

        result = precond._inv @ test_matrix_batch

        # Check shape, device and precision
        assert result.shape == test_matrix_batch.shape
        assert result.device == device
        assert result.dtype == precision


class TestNewtonConsistency:
    """Tests for mathematical consistency of Newton preconditioner."""

    def test_inverse_consistency(
        self, test_matrix, test_vector, newton_config, device, precision, tol
    ):
        """Test P @ (P^-1 @ x) ≈ x."""
        precond = Newton(newton_config)
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

    def test_spd_property(self, test_matrix, newton_config, device, precision, tol):
        """Test that the preconditioner is symmetric positive definite."""
        precond = Newton(newton_config)
        precond._update(test_matrix, device)

        # Test symmetry: construct explicit matrix representation and check symmetry
        n = test_matrix.shape[0]
        P_explicit = precond @ torch.eye(n, device=device, dtype=precision)

        # Check symmetry: P should equal its transpose
        assert torch.allclose(P_explicit, P_explicit.T, **tol)

        # Check positive definiteness: all eigenvalues should be positive
        eigvals = torch.linalg.eigvalsh(P_explicit)
        min_eigval = torch.min(eigvals)
        assert (
            min_eigval > 0
        ), f"Minimum eigenvalue should be positive, but got {min_eigval}"

        # Additionally check quadratic form for a random vector
        x = torch.randn(n, device=device, dtype=precision)
        x = x / torch.norm(x)  # Normalize
        Px = precond @ x
        quad_form = torch.dot(x, Px)
        assert quad_form > 0


class TestNewtonWithLinOp:
    """Tests for Newton preconditioner with linear operators."""

    def test_update_with_linop(
        self, symmetric_linop, newton_config, device, precision, tol
    ):
        """Test updating the Newton preconditioner with a symmetric linear operator."""
        precond = Newton(newton_config)
        precond._update(symmetric_linop, device)

        # After update, L should exist
        assert precond.L is not None
        assert precond.L.device == device
        assert precond.L.dtype == precision

        # L should be lower triangular
        assert torch.allclose(precond.L, torch.tril(precond.L), **tol)

        # Check dimensions
        assert precond.L.shape == symmetric_linop.shape

        # Test that the linop is handled correctly by comparing with tensor approach
        # Create a tensor version of the same operation
        test_matrix = symmetric_linop @ torch.eye(
            symmetric_linop.shape[0], device=device, dtype=precision
        )

        precond_tensor = Newton(newton_config)
        precond_tensor._update(test_matrix, device)

        # The Cholesky factors might differ by a permutation, so we check
        # that L@L.T is close to the original matrix for both methods
        linop_reconstruct = precond.L @ precond.L.T
        tensor_reconstruct = precond_tensor.L @ precond_tensor.L.T

        # They should be close within tolerance
        assert torch.allclose(linop_reconstruct, tensor_reconstruct, **tol)
