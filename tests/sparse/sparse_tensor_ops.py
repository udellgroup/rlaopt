import pytest
import numpy as np
import scipy.sparse as sp
import torch

from rlaopt.sparse import SparseCSRTensor


def get_available_devices():
    """Return a list of available devices to test."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    return devices


def format_device_name(device_str):
    """Format device name for display."""
    if device_str == "cpu":
        return "CPU"
    else:
        gpu_id = device_str.split(":")[-1]
        return f"GPU {gpu_id}"


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-6},
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


# Fixture for test data preparation with different precisions
@pytest.fixture(scope="module")
def sparse_data_dict():
    """Generate test data once and cache different precision versions."""
    n = 10000
    d = 30000
    density = 1e-4

    # Generate base data in high precision
    base_data = sp.random_array((n, d), density=density, format="csr", dtype=np.float64)

    # Create dictionary with both precision versions
    return {
        torch.float32: base_data.astype(np.float32),
        torch.float64: base_data,  # Already float64, no conversion needed
    }


# Fixture for high-precision reference data
@pytest.fixture(scope="module")
def reference_data(sparse_data_dict):
    """Get high-precision reference data (always float64)."""
    # Return the float64 version of our test data
    return sparse_data_dict[torch.float64]


# Fixture for properly typed sparse_data based on precision
@pytest.fixture
def sparse_data(sparse_data_dict, precision):
    """Return sparse data with appropriate precision."""
    return sparse_data_dict[precision]


# Fixture for tolerance values
@pytest.fixture
def tol(precision):
    """Return appropriate tolerance values for the current precision."""
    return TOLERANCES[precision]


# Fixture for device parameters
@pytest.fixture(params=get_available_devices())
def device(request):
    """Parameterized fixture for testing on different devices."""
    return request.param


# Fixtures for different precisions
@pytest.fixture(params=[torch.float32, torch.float64])
def precision(request):
    """Parameterized fixture for testing with different precision."""
    torch.set_default_dtype(request.param)  # Set default dtype for torch
    return request.param


# Fixture to create the sparse tensor for testing
@pytest.fixture
def sparse_tensor(sparse_data, device):
    """Create a sparse tensor with appropriate precision and device."""
    # Create the sparse tensor
    return SparseCSRTensor(data=sparse_data, device=device)


# Test cases
def test_row_slicing(sparse_tensor, device, tol):
    """Test for CSR row slicing."""
    idx_size = min(1024, sparse_tensor.shape[0])
    idx = torch.tensor(np.arange(idx_size), dtype=torch.int64, device=device)
    b = torch.randn(sparse_tensor.shape[1], 10, device=device)

    # Operation and reference
    sliced_result = sparse_tensor[idx] @ b
    reference_result = (sparse_tensor @ b)[idx]

    # Verify results
    assert torch.allclose(sliced_result, reference_result, **tol)


def test_csc_matvec(sparse_tensor, reference_data, device, tol):
    """Test for CSC matrix-vector multiply."""
    # Create random vector
    d = torch.randn(sparse_tensor.shape[0], device=device)

    # Get numpy versions for reference calc (high precision)
    d_np_64 = d.cpu().numpy().astype(np.float64)

    # Operation using the tensor with the current precision
    result = sparse_tensor.T @ d

    # Reference calculation using high precision
    reference = reference_data.T @ d_np_64

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor (always float64)
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


@pytest.mark.parametrize("cols", [1, 32, 128])
def test_csc_matmat(sparse_tensor, reference_data, device, tol, cols):
    """Test for CSC matrix-matrix multiply with different column sizes."""
    # Skip large matrix multiplication on CPU for performance reasons
    if device == "cpu" and cols > 32:
        pytest.skip("Skipping large matrix multiplication on CPU")

    # Create random matrix for multiplication
    D = torch.randn(sparse_tensor.shape[0], cols, device=device)

    # Get numpy versions for reference calc (high precision)
    D_np_64 = D.cpu().numpy().astype(np.float64)

    # Operation using the tensor with the current precision
    result = sparse_tensor.T @ D

    # Reference calculation using high precision
    reference = reference_data.T @ D_np_64

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor (always float64)
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


# # Performance testing - can be marked to run separately if needed
# @pytest.mark.performance
# def test_csc_matmat_performance(sparse_tensor, sparse_data, device):
#     """Test performance of CSC matrix-matrix multiply."""
#     # Skip on CI or when running basic tests
#     if (
#         os.environ.get("CI") == "true"
#         or os.environ.get("SKIP_PERFORMANCE_TESTS") == "true"
#     ):
#         pytest.skip("Skipping performance test in CI environment")

#     cols = 32
#     D = torch.randn(sparse_tensor.shape[0], cols, device=device)
#     D_np = D.cpu().numpy()

#     def operation():
#         return sparse_tensor.T @ D

#     def reference_operation():
#         return sparse_data.T @ D_np

#     elapsed, ref_elapsed, is_correct = time_and_verify_operation(
#         operation, reference_operation, device
#     )

#     # Print performance results
#     print(f"\nPerformance test on {format_device_name(device)}:")
#     print(f"  CSC Matrix-Matrix Multiply ({cols} columns):")
#     print(f"    Time (extension): {elapsed:.6f}s")
#     print(f"    Time (scipy): {ref_elapsed:.6f}s")
#     print(f"    Speedup: {ref_elapsed/elapsed:.2f}x")

#     # Verify results are correct
#     assert is_correct

#     # On GPU, we generally expect our implementation to be faster
#     if "cuda" in device:
#         assert (
#             elapsed < ref_elapsed
#         ), "GPU implementation should be faster than scipy reference"
