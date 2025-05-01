import pytest
import torch

from rlaopt.linops import TwoSidedLinOp
from rlaopt.linops.base import _BaseLinOp
from rlaopt.kernels import DistributedRBFLinOp, KernelConfig

from tests.kernels.utils import compute_kernel_matrix, rbf_kernel


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


class TestDistributedKernelLinOps:
    """Tests for distributed kernel linear operators with full kernel computation."""

    def test_initialization(self, test_matrices, kernel_config, devices):
        """Test initialization of distributed kernel operators."""
        kernel = DistributedRBFLinOp(
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

    def test_matmul_and_transpose(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_matmul_matrix,
        tol,
        devices,
    ):
        """Test matrix-vector multiplication and transpose with
        distributed kernel operators."""
        kernel = DistributedRBFLinOp(
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
                rbf_kernel,
            )

            # ===== Test 1: Matrix-Vector Multiplication =====
            dist_result = kernel @ test_matmul_vector
            expected_result = K_looped @ test_matmul_vector
            assert torch.allclose(dist_result, expected_result, **tol)

            # ===== Test 2: Matrix-Matrix Multiplication =====
            dist_result_mat = kernel @ test_matmul_matrix
            expected_result_mat = K_looped @ test_matmul_matrix
            assert torch.allclose(dist_result_mat, expected_result_mat, **tol)

            # ===== Test 3: Right Multiplication (Transpose) =====
            # Create a vector of matching size for transpose multiplication
            rhs_vector = torch.randn(
                test_matrices["A1"].shape[0],
                device=test_matrices["A1"].device,
                dtype=test_matrices["A1"].dtype,
            )

            trans_result = rhs_vector @ kernel
            expected_trans_result = rhs_vector @ K_looped
            assert torch.allclose(trans_result, expected_trans_result, **tol)

            # ===== Test 4: Explicit Transpose =====
            transposed_kernel = kernel.T
            trans_result2 = transposed_kernel @ rhs_vector
            expected_trans_result2 = K_looped.T @ rhs_vector
            assert torch.allclose(trans_result2, expected_trans_result2, **tol)

        finally:
            # Ensure proper cleanup
            kernel.shutdown()

    def test_oracles(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_blk_matmul_vector,
        test_blk,
        tol,
        devices,
    ):
        """Test row and block oracles of distributed kernel operators."""
        kernel = DistributedRBFLinOp(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=True,
        )

        try:
            # ===== Test 1: Row Oracle =====
            # Compute non-distributed row oracle for comparison
            X_row = test_matrices["A1"][test_blk]
            Y_row = test_matrices["A2"]
            K_row_looped = compute_kernel_matrix(
                X_row,
                Y_row,
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A1"].dtype,
                rbf_kernel,
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

            # ===== Test 2: Block Oracle =====
            # Compute non-distributed block oracle for comparison
            X_blk = test_matrices["A1"][test_blk]
            Y_blk = test_matrices["A2"][test_blk]
            K_blk_looped = compute_kernel_matrix(
                X_blk,
                Y_blk,
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A1"].dtype,
                rbf_kernel,
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


class TestDistributedKernelOraclesOnly:
    """Tests for distributed kernel linear operators with
    oracles only (no full kernel)."""

    def test_initialization(self, test_matrices, kernel_config, devices):
        """Test initialization of distributed kernel operators in oracle-only mode."""
        kernel = DistributedRBFLinOp(
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

    def test_oracles_without_full_kernel(
        self,
        test_matrices,
        kernel_config,
        test_matmul_vector,
        test_blk_matmul_vector,
        test_blk,
        tol,
        devices,
    ):
        """Test row and block oracles work correctly without
        computing the full kernel."""
        kernel = DistributedRBFLinOp(
            test_matrices["A1"],
            test_matrices["A2"],
            kernel_config=kernel_config,
            devices=devices,
            use_full_kernel=False,  # Don't compute the full kernel
        )

        try:
            # ===== Test 1: Row Oracle =====
            # Compute non-distributed row oracle for comparison
            X_row = test_matrices["A1"][test_blk]
            Y_row = test_matrices["A2"]
            K_row_looped = compute_kernel_matrix(
                X_row,
                Y_row,
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A1"].dtype,
                rbf_kernel,
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

            # ===== Test 2: Block Oracle =====
            # Compute non-distributed block oracle for comparison
            X_blk = test_matrices["A1"][test_blk]
            Y_blk = test_matrices["A2"][test_blk]
            K_blk_looped = compute_kernel_matrix(
                X_blk,
                Y_blk,
                kernel_config,
                test_matrices["A1"].device,
                test_matrices["A1"].dtype,
                rbf_kernel,
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
