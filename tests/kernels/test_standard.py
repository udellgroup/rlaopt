import pytest
import torch

from rlaopt.linops import LinOp, SymmetricLinOp
from rlaopt.kernels import RBFLinOp, LaplaceLinOp


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
    """Create a test matrix."""
    return torch.randn(10, 3, device=device, dtype=precision)


@pytest.fixture
def test_blk():
    """Create a block for testing row and block oracle."""
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture
def test_matmul_vector(test_matrix):
    """Create a test vector for matmul."""
    return torch.randn(
        test_matrix.shape[0], device=test_matrix.device, dtype=test_matrix.dtype
    )


@pytest.fixture
def test_matmul_matrix(test_matrix):
    """Create a test matrix for matmul."""
    return torch.randn(
        test_matrix.shape[0], 2, device=test_matrix.device, dtype=test_matrix.dtype
    )


@pytest.fixture
def test_blk_matmul_vector(test_blk, test_matrix):
    """Create a test vector for block matmul."""
    return torch.randn(
        test_blk.shape[0], device=test_matrix.device, dtype=test_matrix.dtype
    )


@pytest.fixture
def test_blk_matmul_matrix(test_blk, test_matrix):
    """Create a test matrix for block matmul."""
    return torch.randn(
        test_blk.shape[0], 2, device=test_matrix.device, dtype=test_matrix.dtype
    )


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


@pytest.fixture
def tol(precision):
    """Return appropriate tolerance values for the current precision."""
    return TOLERANCES[precision]


@pytest.fixture(
    params=[
        {"name": "scalar", "value": 1.0},
        {"name": "tensor", "value": None},  # Will be filled dynamically
    ],
    ids=["scalar", "tensor"],
)
def lengthscale_param(request, device, precision):
    """Parameterized fixture for different lengthscale types."""
    if request.param["name"] == "scalar":
        return {"lengthscale": request.param["value"]}
    else:
        # Create tensor lengthscale with correct device and precision
        return {
            "lengthscale": torch.tensor([1.0, 2.0, 3.0], device=device, dtype=precision)
        }


# Helper functions to compute kernel matrices
def compute_rbf_kernel_matrix(X, Y, lengthscale, device, dtype):
    """
    Compute RBF kernel matrix between X and Y.

    Args:
        X: First set of points (n_x, d)
        Y: Second set of points (n_y, d)
        lengthscale: Lengthscale parameter (scalar or tensor)
        device: Device for the output
        dtype: Data type for the output

    Returns:
        K: Kernel matrix (n_x, n_y)
    """
    K = torch.zeros((X.shape[0], Y.shape[0]), device=device, dtype=dtype)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i, j] = torch.exp(-1 / 2 * torch.sum(((X[i] - Y[j]) / lengthscale) ** 2))
    return K


def compute_laplace_kernel_matrix(X, Y, lengthscale, device, dtype):
    """
    Compute Laplace kernel matrix between X and Y.

    Args:
        X: First set of points (n_x, d)
        Y: Second set of points (n_y, d)
        lengthscale: Lengthscale parameter (scalar or tensor)
        device: Device for the output
        dtype: Data type for the output

    Returns:
        K: Kernel matrix (n_x, n_y)
    """
    K = torch.zeros((X.shape[0], Y.shape[0]), device=device, dtype=dtype)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i, j] = torch.exp(-torch.sum(torch.abs(X[i] - Y[j]) / lengthscale))
    return K


# Define kernel configurations for parameterized testing
KERNEL_CONFIGS = [
    {
        "class": RBFLinOp,
        "name": "rbf",
        "compute_kernel": compute_rbf_kernel_matrix,
    },
    {
        "class": LaplaceLinOp,
        "name": "laplace",
        "compute_kernel": compute_laplace_kernel_matrix,
    },
]


@pytest.fixture(
    params=KERNEL_CONFIGS, ids=[config["name"] for config in KERNEL_CONFIGS]
)
def kernel_config(request):
    """Parameterized fixture for different kernel types."""
    return request.param


class TestKernelLinOps:
    def test_initialization(self, test_matrix, lengthscale_param, kernel_config):
        """Test initialization of kernel linear operators."""
        kernel_class = kernel_config["class"]
        kernel = kernel_class(test_matrix, kernel_params=lengthscale_param)
        assert kernel.A.shape == test_matrix.shape
        assert kernel.kernel_params == lengthscale_param

    def test_row_and_block_oracle(
        self,
        test_matrix,
        lengthscale_param,
        test_matmul_vector,
        test_matmul_matrix,
        test_blk,
        test_blk_matmul_vector,
        test_blk_matmul_matrix,
        tol,
        kernel_config,
    ):
        """Test row and block oracles of kernel linear operators."""
        kernel_class = kernel_config["class"]
        compute_kernel = kernel_config["compute_kernel"]
        lengthscale = lengthscale_param["lengthscale"]

        kernel = kernel_class(test_matrix, kernel_params=lengthscale_param)

        # Test row oracle
        X_row = test_matrix[test_blk]
        Y_row = test_matrix
        K_row_looped = compute_kernel(
            X_row, Y_row, lengthscale, kernel.device, kernel.dtype
        )

        row_lin_op = kernel.row_oracle(test_blk)
        assert isinstance(row_lin_op, LinOp)
        assert row_lin_op.shape == (test_blk.shape[0], test_matrix.shape[0])
        assert row_lin_op.device == kernel.device
        assert row_lin_op.dtype == kernel.dtype
        assert torch.allclose(
            row_lin_op @ test_matmul_vector, K_row_looped @ test_matmul_vector, **tol
        )
        assert torch.allclose(
            row_lin_op @ test_matmul_matrix, K_row_looped @ test_matmul_matrix, **tol
        )

        # Test block oracle
        X_blk = test_matrix[test_blk]
        Y_blk = test_matrix[test_blk]
        K_blk_looped = compute_kernel(
            X_blk, Y_blk, lengthscale, kernel.device, kernel.dtype
        )

        block_lin_op = kernel.blk_oracle(test_blk)
        assert isinstance(block_lin_op, SymmetricLinOp)
        assert block_lin_op.shape == (test_blk.shape[0], test_blk.shape[0])
        assert block_lin_op.device == kernel.device
        assert block_lin_op.dtype == kernel.dtype
        assert torch.allclose(
            block_lin_op @ test_blk_matmul_vector,
            K_blk_looped @ test_blk_matmul_vector,
            **tol
        )
        assert torch.allclose(
            block_lin_op @ test_blk_matmul_matrix,
            K_blk_looped @ test_blk_matmul_matrix,
            **tol
        )

    def test_matmul(
        self,
        test_matrix,
        lengthscale_param,
        test_matmul_vector,
        test_matmul_matrix,
        tol,
        kernel_config,
    ):
        """Test matmul of kernel linear operators."""
        kernel_class = kernel_config["class"]
        compute_kernel = kernel_config["compute_kernel"]
        lengthscale = lengthscale_param["lengthscale"]

        kernel = kernel_class(test_matrix, kernel_params=lengthscale_param)

        # Compute full kernel matrix using helper function
        K_looped = compute_kernel(
            test_matrix, test_matrix, lengthscale, kernel.device, kernel.dtype
        )

        assert torch.allclose(
            kernel @ test_matmul_vector, K_looped @ test_matmul_vector, **tol
        )
        assert torch.allclose(
            kernel @ test_matmul_matrix, K_looped @ test_matmul_matrix, **tol
        )
