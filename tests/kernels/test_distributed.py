import pytest
import torch

from rlaopt.linops import TwoSidedLinOp
from rlaopt.linops.base import _BaseLinOp
from rlaopt.kernels import (
    DistributedRBFLinOp,
    DistributedLaplaceLinOp,
    DistributedMatern12LinOp,
    DistributedMatern32LinOp,
    DistributedMatern52LinOp,
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


@pytest.fixture
def devices():
    """Return a set of available devices for distributed testing."""
    devices = {torch.device("cpu")}
    if torch.cuda.is_available():
        devices.add(torch.device("cuda:0"))
    return devices


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def precision(request):
    """Parameterized fixture for testing with different precision."""
    return request.param


@pytest.fixture
def test_matrices(device, precision):
    """Create test matrices."""
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


@pytest.fixture
def kernel_config(device, precision):
    """Create a kernel configuration with scaling."""
    const_scaling = 2.0  # Using a non-default value to test scaling
    return KernelConfig(const_scaling=const_scaling, lengthscale=1.0)


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


@pytest.fixture
def tol(precision):
    """Return appropriate tolerance values for the current precision."""
    return TOLERANCES[precision]


# Define kernel configurations for parameterized testing
DISTRIBUTED_KERNEL_PARAMETERIZATIONS = [
    {
        "class": DistributedRBFLinOp,
        "name": "distributed_rbf",
        "kernel_func": rbf_kernel,
    },
    {
        "class": DistributedLaplaceLinOp,
        "name": "distributed_laplace",
        "kernel_func": laplace_kernel,
    },
    {
        "class": DistributedMatern12LinOp,
        "name": "distributed_matern12",
        "kernel_func": matern12_kernel,
    },
    {
        "class": DistributedMatern32LinOp,
        "name": "distributed_matern32",
        "kernel_func": matern32_kernel,
    },
    {
        "class": DistributedMatern52LinOp,
        "name": "distributed_matern52",
        "kernel_func": matern52_kernel,
    },
]


@pytest.fixture(
    params=DISTRIBUTED_KERNEL_PARAMETERIZATIONS,
    ids=[config["name"] for config in DISTRIBUTED_KERNEL_PARAMETERIZATIONS],
)
def distributed_kernel_parameterization(request):
    """Parameterized fixture for different distributed kernel types."""
    return request.param


class TestDistributedKernelLinOps:
    """Tests for distributed kernel linear operators with full kernel computation."""

    def test_initialization(
        self, test_matrices, kernel_config, devices, distributed_kernel_parameterization
    ):
        """Test initialization of distributed kernel operators."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # Check basic properties
            assert kernel.A1.shape == test_matrices["A1"].shape
            assert kernel.A2.shape == test_matrices["A2"].shape
            assert kernel.kernel_config == kernel_config
            assert kernel.dtype == test_matrices["A1"].dtype

            # Check distributed properties
            assert kernel.shape == torch.Size(
                (test_matrices["A1"].shape[0], test_matrices["A2"].shape[0])
            )
            assert len(kernel.devices) == len(devices)
            assert kernel._scaling == kernel_config.const_scaling

            # Check row chunks
            assert len(kernel.A1_row_chunks) == len(devices)
            total_rows = sum(chunk.shape[0] for chunk in kernel.A1_row_chunks)
            assert total_rows == test_matrices["A1"].shape[0]

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_matmul(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_matmul_matrix,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test matrix-vector multiplication with distributed kernel operators."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel_func = distributed_kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # Compute non-distributed kernel for comparison
            K_looped = compute_kernel_matrix(
                test_matrices["A1"],
                test_matrices["A2"],
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A2"].dtype,
                kernel_func,
            )

            # Test vector multiplication
            dist_result = kernel @ test_matmul_vector
            expected_result = K_looped @ test_matmul_vector
            assert torch.allclose(dist_result, expected_result, **tol)

            # Test matrix multiplication
            dist_result_mat = kernel @ test_matmul_matrix
            expected_result_mat = K_looped @ test_matmul_matrix
            assert torch.allclose(dist_result_mat, expected_result_mat, **tol)

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_transpose(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test transpose of distributed kernel operators."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel_func = distributed_kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # Compute non-distributed kernel for comparison
            K_looped = compute_kernel_matrix(
                test_matrices["A1"],
                test_matrices["A2"],
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A2"].dtype,
                kernel_func,
            )

            # Create a vector of matching size for transpose multiplication
            rhs_vector = torch.randn(
                test_matrices["A1"].shape[0],
                device=test_matrices["A1"].device,
                dtype=test_matrices["A1"].dtype,
            )

            # Test transpose multiplication
            transposed_kernel = kernel.T
            trans_result = rhs_vector @ kernel
            expected_trans_result = rhs_vector @ K_looped

            assert torch.allclose(trans_result, expected_trans_result, **tol)

            # Also test the transposed kernel directly
            trans_result2 = transposed_kernel @ rhs_vector
            expected_trans_result2 = K_looped.T @ rhs_vector

            assert torch.allclose(trans_result2, expected_trans_result2, **tol)

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_row_oracle(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_blk,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test row oracle of distributed kernel operators."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel_func = distributed_kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # Compute non-distributed row oracle for comparison
            X_row = test_matrices["A1"][test_blk]
            Y_row = test_matrices["A2"]
            K_row_looped = compute_kernel_matrix(
                X_row,
                Y_row,
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A1"].dtype,
                kernel_func,
            )

            # Get row oracle from distributed kernel
            row_oracle = kernel.row_oracle(test_blk)

            try:
                # Verify properties
                assert isinstance(row_oracle, _BaseLinOp)
                assert row_oracle.shape == (
                    test_blk.shape[0],
                    test_matrices["A2"].shape[0],
                )

                # Test multiplication
                oracle_result = row_oracle @ test_matmul_vector
                expected_result = K_row_looped @ test_matmul_vector

                assert torch.allclose(oracle_result, expected_result, **tol)

            finally:
                # Ensure proper cleanup of oracle
                if hasattr(row_oracle, "shutdown"):
                    row_oracle.shutdown()

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_blk_oracle(
        self,
        test_matrices,
        kernel_config,
        test_blk_matmul_vector,
        test_blk,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test block oracle of distributed kernel operators."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel_func = distributed_kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # Compute non-distributed block oracle for comparison
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

            # Get block oracle from distributed kernel
            blk_oracle = kernel.blk_oracle(test_blk)

            try:
                # Verify properties
                assert isinstance(blk_oracle, _BaseLinOp)
                assert blk_oracle.shape == (test_blk.shape[0], test_blk.shape[0])

                # Test multiplication
                oracle_result = blk_oracle @ test_blk_matmul_vector
                expected_result = K_blk_looped @ test_blk_matmul_vector

                assert torch.allclose(oracle_result, expected_result, **tol)

            finally:
                # Ensure proper cleanup of oracle
                if hasattr(blk_oracle, "shutdown"):
                    blk_oracle.shutdown()

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_scaling_in_matmul(
        self,
        test_matrices,
        test_matmul_vector,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test that scaling is correctly applied in matrix-vector multiplication."""
        kernel_class = distributed_kernel_parameterization["class"]

        # Create two configs with different scaling factors
        scale1 = 1.0
        scale2 = 3.5  # Different from default to test scaling is working

        config1 = KernelConfig(const_scaling=scale1, lengthscale=1.0)
        config2 = KernelConfig(const_scaling=scale2, lengthscale=1.0)

        kernel1 = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=config1,
            devices=devices,
            use_full_kernel=True,
        )

        kernel2 = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=config2,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # Perform matrix-vector multiplication with both kernels
            result1 = kernel1 @ test_matmul_vector
            result2 = kernel2 @ test_matmul_vector

            # The ratio of results should match the ratio of scaling factors
            expected_ratio = scale2 / scale1
            actual_ratio = result2 / result1

            # Check that ratio is consistent
            assert torch.allclose(
                actual_ratio, torch.tensor(expected_ratio, device=result1.device), **tol
            )

        finally:
            # Ensure proper cleanup
            kernel1.shutdown()
            kernel2.shutdown()


class TestDistributedKernelOraclesOnly:
    """Tests for distributed kernel linear operators
    with oracles only (no full kernel)."""

    def test_initialization(
        self, test_matrices, kernel_config, devices, distributed_kernel_parameterization
    ):
        """Test initialization of distributed kernel operators in oracle-only mode."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=False,  # Don't compute the full kernel
        )

        try:
            # Check basic properties
            assert kernel.A1.shape == test_matrices["A1"].shape
            assert kernel.A2.shape == test_matrices["A2"].shape
            assert kernel.kernel_config == kernel_config
            assert kernel.dtype == test_matrices["A1"].dtype

            # Check that kernel uses fake operators
            for op in kernel.kernel_ops:
                # The operator should be a TwoSidedLinOp with placeholder methods
                assert isinstance(op, TwoSidedLinOp)

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_row_oracle_without_full_kernel(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_blk,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test row oracle works correctly without computing the full kernel."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel_func = distributed_kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=False,  # Don't compute the full kernel
        )

        try:
            # Compute non-distributed row oracle for comparison
            X_row = test_matrices["A1"][test_blk]
            Y_row = test_matrices["A2"]
            K_row_looped = compute_kernel_matrix(
                X_row,
                Y_row,
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A1"].dtype,
                kernel_func,
            )

            # Get row oracle from distributed kernel
            row_oracle = kernel.row_oracle(test_blk)

            try:
                # Verify properties
                assert isinstance(row_oracle, _BaseLinOp)
                assert row_oracle.shape == (
                    test_blk.shape[0],
                    test_matrices["A2"].shape[0],
                )

                # Test multiplication
                oracle_result = row_oracle @ test_matmul_vector
                expected_result = K_row_looped @ test_matmul_vector

                assert torch.allclose(oracle_result, expected_result, **tol)

            finally:
                # Ensure proper cleanup of oracle
                if hasattr(row_oracle, "shutdown"):
                    row_oracle.shutdown()

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_blk_oracle_without_full_kernel(
        self,
        test_matrices,
        kernel_config,
        test_blk_matmul_vector,
        test_blk,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test block oracle works correctly without computing the full kernel."""
        kernel_class = distributed_kernel_parameterization["class"]
        kernel_func = distributed_kernel_parameterization["kernel_func"]

        kernel = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=False,  # Don't compute the full kernel
        )

        try:
            # Compute non-distributed block oracle for comparison
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

            # Get block oracle from distributed kernel
            blk_oracle = kernel.blk_oracle(test_blk)

            try:
                # Verify properties
                assert isinstance(blk_oracle, _BaseLinOp)
                assert blk_oracle.shape == (test_blk.shape[0], test_blk.shape[0])

                # Test multiplication
                oracle_result = blk_oracle @ test_blk_matmul_vector
                expected_result = K_blk_looped @ test_blk_matmul_vector

                assert torch.allclose(oracle_result, expected_result, **tol)

            finally:
                # Ensure proper cleanup of oracle
                if hasattr(blk_oracle, "shutdown"):
                    blk_oracle.shutdown()

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_scaling_in_oracles(
        self,
        test_matrices,
        test_blk,
        test_blk_matmul_vector,
        tol,
        devices,
        distributed_kernel_parameterization,
    ):
        """Test that scaling is correctly applied in oracles
        when not using full kernel."""
        kernel_class = distributed_kernel_parameterization["class"]

        # Create two configs with different scaling factors
        scale1 = 1.0
        scale2 = 3.5  # Different from default to test scaling is working

        config1 = KernelConfig(const_scaling=scale1, lengthscale=1.0)
        config2 = KernelConfig(const_scaling=scale2, lengthscale=1.0)

        kernel1 = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=config1,
            devices=devices,
            use_full_kernel=False,
        )

        kernel2 = kernel_class(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=config2,
            devices=devices,
            use_full_kernel=False,
        )

        try:
            # Get block oracles
            blk_oracle1 = kernel1.blk_oracle(test_blk)
            blk_oracle2 = kernel2.blk_oracle(test_blk)

            try:
                # Perform matrix-vector multiplication with both oracles
                result1 = blk_oracle1 @ test_blk_matmul_vector
                result2 = blk_oracle2 @ test_blk_matmul_vector

                # The ratio of results should match the ratio of scaling factors
                expected_ratio = scale2 / scale1
                actual_ratio = result2 / result1

                # Check that ratio is consistent
                assert torch.allclose(
                    actual_ratio,
                    torch.tensor(expected_ratio, device=result1.device),
                    **tol
                )

            finally:
                # Ensure proper cleanup of oracles
                if hasattr(blk_oracle1, "shutdown"):
                    blk_oracle1.shutdown()
                if hasattr(blk_oracle2, "shutdown"):
                    blk_oracle2.shutdown()

        finally:
            # Ensure proper cleanup
            kernel1.shutdown()
            kernel2.shutdown()
