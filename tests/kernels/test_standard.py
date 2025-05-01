import pytest
import torch

from rlaopt.linops.base import _BaseLinOp
from rlaopt.kernels import (
    RBFLinOp,
    LaplaceLinOp,
    Matern12LinOp,
    Matern32LinOp,
    Matern52LinOp,
    KernelConfig,
)

from tests.kernels.utils import (
    compute_kernel_matrix,
    rbf_kernel,
    laplace_kernel,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
)


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
def test_matrices(device, precision):
    """Create a test matrix."""
    return {
        "A1": torch.randn(10, 3, device=device, dtype=precision),
        "A2": torch.randn(5, 3, device=device, dtype=precision),
    }


@pytest.fixture
def test_blk():
    """Create a block for testing row and block oracle."""
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture
def test_matmul_vector(test_matrices):
    """Create a test vector for matmul."""
    return torch.randn(
        test_matrices["A2"].shape[0],
        device=test_matrices["A2"].device,
        dtype=test_matrices["A2"].dtype,
    )


@pytest.fixture
def test_matmul_matrix(test_matrices):
    """Create a test matrix for matmul."""
    return torch.randn(
        test_matrices["A2"].shape[0],
        2,
        device=test_matrices["A2"].device,
        dtype=test_matrices["A2"].dtype,
    )


@pytest.fixture
def test_blk_matmul_vector(test_blk, test_matrices):
    """Create a test vector for block matmul."""
    return torch.randn(
        test_blk.shape[0],
        device=test_matrices["A2"].device,
        dtype=test_matrices["A2"].dtype,
    )


@pytest.fixture
def test_blk_matmul_matrix(test_blk, test_matrices):
    """Create a test matrix for block matmul."""
    return torch.randn(
        test_blk.shape[0],
        2,
        device=test_matrices["A2"].device,
        dtype=test_matrices["A2"].dtype,
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
def kernel_config(request, device, precision):
    const_scaling = 2.0
    """Parameterized fixture for different lengthscale types."""
    if request.param["name"] == "scalar":
        return KernelConfig(
            const_scaling=const_scaling, lengthscale=request.param["value"]
        )
    else:
        lengthscale = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=precision)
        return KernelConfig(const_scaling=const_scaling, lengthscale=lengthscale)


# Define kernel configurations for parameterized testing
KERNEL_PARAMETERIZATIONS = [
    {
        "class": RBFLinOp,
        "name": "rbf",
        "kernel_func": rbf_kernel,
    },
    {
        "class": LaplaceLinOp,
        "name": "laplace",
        "kernel_func": laplace_kernel,
    },
    {
        "class": Matern12LinOp,
        "name": "matern12",
        "kernel_func": matern12_kernel,
    },
    {
        "class": Matern32LinOp,
        "name": "matern32",
        "kernel_func": matern32_kernel,
    },
    {
        "class": Matern52LinOp,
        "name": "matern52",
        "kernel_func": matern52_kernel,
    },
]


@pytest.fixture(
    params=KERNEL_PARAMETERIZATIONS,
    ids=[config["name"] for config in KERNEL_PARAMETERIZATIONS],
)
def kernel_parameterization(request):
    """Parameterized fixture for different kernel types."""
    return request.param


class TestKernelLinOps:
    def test_initialization(
        self, test_matrices, kernel_config, kernel_parameterization
    ):
        """Test initialization of kernel linear operators."""
        kernel_class = kernel_parameterization["class"]
        kernel = kernel_class(
            test_matrices["A1"], test_matrices["A2"], kernel_config=kernel_config
        )
        assert kernel.A1.shape == test_matrices["A1"].shape
        assert kernel.A2.shape == test_matrices["A2"].shape
        assert kernel.kernel_config == kernel_config
        assert kernel.dtype == test_matrices["A1"].dtype

    def test_row_and_block_oracle(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_matmul_matrix,
        test_blk,
        test_blk_matmul_vector,
        test_blk_matmul_matrix,
        tol,
        kernel_parameterization,
    ):
        """Test row and block oracles of kernel linear operators."""
        kernel_class = kernel_parameterization["class"]
        kernel_func = kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"], test_matrices["A2"], kernel_config=kernel_config
        )

        # Test row oracle
        X_row = test_matrices["A1"][test_blk]
        Y_row = test_matrices["A2"]
        K_row_looped = compute_kernel_matrix(
            X_row, Y_row, kernel_config, kernel.device, kernel.dtype, kernel_func
        )

        row_lin_op = kernel.row_oracle(test_blk)
        assert isinstance(row_lin_op, _BaseLinOp)
        assert row_lin_op.shape == (test_blk.shape[0], test_matrices["A2"].shape[0])
        assert row_lin_op.device == kernel.device
        assert row_lin_op.dtype == kernel.dtype
        assert torch.allclose(
            row_lin_op @ test_matmul_vector, K_row_looped @ test_matmul_vector, **tol
        )
        assert torch.allclose(
            row_lin_op @ test_matmul_matrix, K_row_looped @ test_matmul_matrix, **tol
        )

        # Test block oracle
        X_blk = test_matrices["A1"][test_blk]
        Y_blk = test_matrices["A2"][test_blk]
        K_blk_looped = compute_kernel_matrix(
            X_blk,
            Y_blk,
            kernel_config,
            test_matrices["A1"].device,
            test_matrices["A1"].dtype,
            kernel_func,
        )

        block_lin_op = kernel.blk_oracle(test_blk)
        assert isinstance(block_lin_op, _BaseLinOp)
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
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_matmul_matrix,
        tol,
        kernel_parameterization,
    ):
        """Test matmul of kernel linear operators."""
        kernel_class = kernel_parameterization["class"]
        kernel_func = kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"], test_matrices["A2"], kernel_config=kernel_config
        )

        # Compute full kernel matrix using helper function
        K_looped = compute_kernel_matrix(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config,
            test_matrices["A1"].device,
            test_matrices["A2"].dtype,
            kernel_func,
        )

        assert torch.allclose(
            kernel @ test_matmul_vector, K_looped @ test_matmul_vector, **tol
        )
        assert torch.allclose(
            kernel @ test_matmul_matrix, K_looped @ test_matmul_matrix, **tol
        )
