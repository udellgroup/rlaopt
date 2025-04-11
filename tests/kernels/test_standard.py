import pytest
import torch

from rlaopt.linops import LinOp, SymmetricLinOp
from rlaopt.kernels import RBFLinOp


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
def scalar_lengthscale():
    """Fixture for scalar lengthscale."""
    return {"lengthscale": 1.0}


@pytest.fixture
def tensor_lengthscale(device, precision):
    """Fixture for tensor lengthscale."""
    return {
        "lengthscale": torch.tensor([1.0, 2.0, 3.0], device=device, dtype=precision)
    }


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


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


@pytest.fixture
def tol(precision):
    """Return appropriate tolerance values for the current precision."""
    return TOLERANCES[precision]


class TestRBFLinOp:
    def test_initialization_with_scalar_lengthscale(
        self, test_matrix, scalar_lengthscale
    ):
        """Test initialization of RBFLinOp."""
        kernel = RBFLinOp(test_matrix, kernel_params=scalar_lengthscale)
        assert kernel.A.shape == test_matrix.shape
        assert kernel.kernel_params == scalar_lengthscale

    def test_initialization_with_tensor_lengthscale(
        self, test_matrix, tensor_lengthscale
    ):
        """Test initialization of RBFLinOp with tensor lengthscale."""
        kernel = RBFLinOp(test_matrix, kernel_params=tensor_lengthscale)
        assert kernel.A.shape == test_matrix.shape
        assert kernel.kernel_params == tensor_lengthscale

    def test_row_and_block_oracle_scalar_lengthscale(
        self, test_matrix, scalar_lengthscale, test_blk
    ):
        """Test row and block oracles of RBFLinOp."""
        kernel = RBFLinOp(test_matrix, kernel_params=scalar_lengthscale)

        row_lin_op = kernel.row_oracle(test_blk)
        assert isinstance(row_lin_op, LinOp)
        assert row_lin_op.shape == (test_blk.shape[0], test_matrix.shape[0])
        assert row_lin_op.device == kernel.device
        assert row_lin_op.dtype == kernel.dtype

        block_lin_op = kernel.blk_oracle(test_blk)
        assert isinstance(block_lin_op, SymmetricLinOp)
        assert block_lin_op.shape == (test_blk.shape[0], test_blk.shape[0])
        assert block_lin_op.device == kernel.device
        assert block_lin_op.dtype == kernel.dtype

    def test_row_and_block_oracle_tensor_lengthscale(
        self, test_matrix, tensor_lengthscale, test_blk
    ):
        """Test row and block oracles of RBFLinOp with tensor lengthscale."""
        kernel = RBFLinOp(test_matrix, kernel_params=tensor_lengthscale)

        row_lin_op = kernel.row_oracle(test_blk)
        assert isinstance(row_lin_op, LinOp)
        assert row_lin_op.shape == (test_blk.shape[0], test_matrix.shape[0])
        assert row_lin_op.device == kernel.device
        assert row_lin_op.dtype == kernel.dtype

        block_lin_op = kernel.blk_oracle(test_blk)
        assert isinstance(block_lin_op, SymmetricLinOp)
        assert block_lin_op.shape == (test_blk.shape[0], test_blk.shape[0])
        assert block_lin_op.device == kernel.device
        assert block_lin_op.dtype == kernel.dtype

    def test_matmul_scalar_lengthscale(
        self,
        test_matrix,
        scalar_lengthscale,
        test_matmul_vector,
        test_matmul_matrix,
        tol,
    ):
        """Test matmul of RBFLinOp with scalar lengthscale."""
        lengthscale = scalar_lengthscale["lengthscale"]

        # Form the kernel matrix manually
        K_looped = torch.zeros(
            (test_matrix.shape[0], test_matrix.shape[0]),
            device=test_matrix.device,
            dtype=test_matrix.dtype,
        )
        for i in range(K_looped.shape[0]):
            for j in range(K_looped.shape[1]):
                K_looped[i, j] = torch.exp(
                    -1
                    / 2
                    * torch.sum(((test_matrix[i] - test_matrix[j]) / lengthscale) ** 2)
                )

        kernel = RBFLinOp(test_matrix, kernel_params=scalar_lengthscale)
        assert torch.allclose(
            kernel @ test_matmul_vector, K_looped @ test_matmul_vector, **tol
        )
        assert torch.allclose(
            kernel @ test_matmul_matrix, K_looped @ test_matmul_matrix, **tol
        )

    def test_matmul_tensor_lengthscale(
        self,
        test_matrix,
        tensor_lengthscale,
        test_matmul_vector,
        test_matmul_matrix,
        tol,
    ):
        """Test matmul of RBFLinOp with tensor lengthscale."""
        lengthscale = tensor_lengthscale["lengthscale"]

        # Form the kernel matrix manually
        K_looped = torch.zeros(
            (test_matrix.shape[0], test_matrix.shape[0]),
            device=test_matrix.device,
            dtype=test_matrix.dtype,
        )
        for i in range(K_looped.shape[0]):
            for j in range(K_looped.shape[1]):
                K_looped[i, j] = torch.exp(
                    -1
                    / 2
                    * torch.sum(((test_matrix[i] - test_matrix[j]) / lengthscale) ** 2)
                )

        kernel = RBFLinOp(test_matrix, kernel_params=tensor_lengthscale)
        assert torch.allclose(
            kernel @ test_matmul_vector, K_looped @ test_matmul_vector, **tol
        )
        assert torch.allclose(
            kernel @ test_matmul_matrix, K_looped @ test_matmul_matrix, **tol
        )
