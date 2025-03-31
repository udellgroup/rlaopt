import time
import os  # noqa: F401
from typing import Callable, Tuple

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


def load_sparse_matrix(file_path, precision):
    """Load sparse matrix from npz file."""
    return sp.load_npz(file_path).astype(precision)


def time_and_verify_operation(
    operation: Callable,
    reference_operation: Callable,
    device_str: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[float, float, bool]:
    """Perform operation, time it, and verify against reference."""
    start = time.time()
    result = operation()
    elapsed = time.time() - start

    ref_start = time.time()
    ref_result = reference_operation()
    ref_elapsed = time.time() - ref_start

    # Convert reference result to torch tensor on the right device if needed
    if isinstance(ref_result, np.ndarray):
        ref_result = torch.tensor(ref_result, device=device_str)

    # Verify results match
    is_correct = torch.allclose(result, ref_result, rtol=rtol, atol=atol)

    return elapsed, ref_elapsed, is_correct


# Fixture for test data preparation
@pytest.fixture(scope="module")
def sparse_data():
    """Fixture to load test data once for all tests."""
    n = 10000
    d = 30000
    density = 1e-4
    return sp.random_array((n, d), density=density, format="csr", dtype=np.float64)


# Fixture for device parameters
@pytest.fixture(params=get_available_devices())
def device(request):
    """Parameterized fixture for testing on different devices."""
    return request.param


# Fixtures for different precisions
@pytest.fixture(params=[torch.float32, torch.float64])
def precision(request):
    """Parameterized fixture for testing with different precision."""
    return request.param


# Fixture to create the sparse tensor for testing
@pytest.fixture
def sparse_tensor(sparse_data, device, precision):
    """Create a sparse tensor with appropriate precision and device."""
    # Set default dtype for torch
    torch.set_default_dtype(precision)

    # Convert data to the right numpy precision
    np_precision = np.float32 if precision == torch.float32 else np.float64
    data = sparse_data.astype(np_precision)

    # Create the sparse tensor
    return SparseCSRTensor(data=data, device=device)


# Test cases
def test_row_slicing(sparse_tensor, device):
    """Test for CSR row slicing."""
    idx_size = min(1024, sparse_tensor.shape[0])
    idx = torch.tensor(np.arange(idx_size), dtype=torch.int64, device=device)
    b = torch.randn(sparse_tensor.shape[1], 10, device=device)

    # Operation and reference
    sliced_result = sparse_tensor[idx] @ b
    reference_result = (sparse_tensor @ b)[idx]

    # Verify results
    assert torch.allclose(sliced_result, reference_result, rtol=1e-5, atol=1e-8)


def test_csc_matvec(sparse_tensor, sparse_data, device):
    """Test for CSC matrix-vector multiply."""
    d = torch.randn(sparse_tensor.shape[0], device=device)
    d_np = d.cpu().numpy()

    # Operation and reference
    result = sparse_tensor.T @ d
    reference = sparse_data.T @ d_np

    # Convert reference to torch tensor
    reference_tensor = torch.tensor(reference, device=device)

    print(f"result.dtype: {result.dtype}")
    print(f"reference_tensor.dtype: {reference_tensor.dtype}")

    # Verify results
    assert torch.allclose(result, reference_tensor, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("cols", [1, 32, 128])
def test_csc_matmat(sparse_tensor, sparse_data, device, cols):
    """Test for CSC matrix-matrix multiply with different column sizes."""
    # Skip large matrix multiplication on CPU for performance reasons
    if device == "cpu" and cols > 32:
        pytest.skip("Skipping large matrix multiplication on CPU")

    D = torch.randn(sparse_tensor.shape[0], cols, device=device)
    D_np = D.cpu().numpy()

    # Operation and reference
    result = sparse_tensor.T @ D
    reference = sparse_data.T @ D_np

    # Convert reference to torch tensor
    reference_tensor = torch.tensor(reference, device=device)

    # Verify results
    assert torch.allclose(result, reference_tensor, rtol=1e-5, atol=1e-8)


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
