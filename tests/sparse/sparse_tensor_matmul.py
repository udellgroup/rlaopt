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


# Dictionary of tolerance values by precision
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-6},
    torch.float64: {"rtol": 1e-8, "atol": 1e-8},
}


# Fixture for test data preparation with different precisions
@pytest.fixture(scope="module")
def sparse_data_dict():
    """Generate test data once and cache different precision versions."""
    n = 2000
    d = 10000
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
@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
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


# -----------------------------------
# Tests for forward matrix multiplication (sparse_tensor @ v)
# -----------------------------------


@pytest.mark.parametrize("shape_type", ["1d", "2d"], ids=["vector", "matrix"])
def test_matmul_csr(sparse_tensor, reference_data, device, tol, shape_type):
    """Test CSR matrix multiplication with different input shapes."""
    n, m = sparse_tensor.shape

    # Create input tensor based on the shape type parameter
    if shape_type == "1d":
        # For 1D input (vector), shape should match width of the matrix
        v = torch.randn(m, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using matmul (sparse_tensor @ vector)
        result = sparse_tensor @ v

        # Reference calculation using high precision
        reference = reference_data @ v_np_64
    else:  # 2d
        # For 2D input (matrix), first dimension must match width,
        # second can be arbitrary
        cols = 10
        v = torch.randn(m, cols, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using matmul (sparse_tensor @ matrix)
        result = sparse_tensor @ v

        # Reference calculation using high precision
        reference = reference_data @ v_np_64

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


@pytest.mark.parametrize("shape_type", ["1d", "2d"], ids=["vector", "matrix"])
def test_matmul_csc(sparse_tensor, reference_data, device, tol, shape_type):
    """Test CSC matrix multiplication (transposed sparse tensor)
    with different input shapes."""
    n, m = sparse_tensor.shape

    # Create input tensor based on the shape type parameter
    if shape_type == "1d":
        # For 1D input with transposed matrix, shape should match height
        v = torch.randn(n, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using matmul (sparse_tensor.T @ vector)
        result = sparse_tensor.T @ v

        # Reference calculation using high precision
        reference = reference_data.T @ v_np_64
    else:  # 2d
        # For 2D input with transposed matrix, first dim must match height,
        # second can be arbitrary
        cols = 10
        v = torch.randn(n, cols, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using matmul (sparse_tensor.T @ matrix)
        result = sparse_tensor.T @ v

        # Reference calculation using high precision
        reference = reference_data.T @ v_np_64

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


def test_matmul_invalid_dims(sparse_tensor):
    """Test that matmul raises ValueError for tensors with invalid dimensions."""
    n, m = sparse_tensor.shape

    # 3D tensor (invalid for matmul)
    invalid_3d = torch.randn(m, 3, 4, device=sparse_tensor.device)

    # Test __matmul__ with 3D tensor
    with pytest.raises(ValueError):
        _ = sparse_tensor @ invalid_3d

    # Also test with transposed matrix
    invalid_3d_t = torch.randn(n, 3, 4, device=sparse_tensor.device)
    with pytest.raises(ValueError):
        _ = sparse_tensor.T @ invalid_3d_t


# -----------------------------------
# Tests for right matrix multiplication (v @ sparse_tensor)
# -----------------------------------


@pytest.mark.parametrize("shape_type", ["1d", "2d"], ids=["vector", "matrix"])
def test_rmatmul_basic(sparse_tensor, reference_data, device, tol, shape_type):
    """Test right matrix multiplication (v @ sparse_tensor)."""
    n, m = sparse_tensor.shape

    # Create input tensor based on the shape type parameter
    if shape_type == "1d":
        # For 1D input (vector), shape should match width of the matrix
        v = torch.randn(n, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using rmatmul (vector @ sparse_tensor)
        result = v @ sparse_tensor

        # Reference calculation using high precision: v @ A = (A.T @ v.T).T = A.T @ v
        # for vector v
        reference = reference_data.T @ v_np_64
    else:  # 2d
        # For 2D input (matrix), first dimension can be arbitrary,
        # second must match width
        batch_size = 5
        v = torch.randn(batch_size, n, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using rmatmul (matrix @ sparse_tensor)
        result = v @ sparse_tensor

        # Reference calculation using high precision: M @ A = (A.T @ M.T).T
        reference = (reference_data.T @ v_np_64.T).T

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor (always float64)
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


@pytest.mark.parametrize("shape_type", ["1d", "2d"], ids=["vector", "matrix"])
def test_rmatmul_transpose(sparse_tensor, reference_data, device, tol, shape_type):
    """Test right matrix multiplication with transpose (v @ sparse_tensor.T)."""
    n, m = sparse_tensor.shape

    # Create input tensor based on the shape type parameter
    if shape_type == "1d":
        # For 1D input with transpose, shape should match height of the original matrix
        v = torch.randn(m, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using rmatmul (vector @ sparse_tensor.T)
        result = v @ sparse_tensor.T

        # Reference calculation using high precision: v @ A.T = A @ v for vector v
        reference = reference_data @ v_np_64
    else:  # 2d
        # For 2D input with transpose, second dimension must
        # match height of original matrix
        batch_size = 5
        v = torch.randn(batch_size, m, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using rmatmul (matrix @ sparse_tensor.T)
        result = v @ sparse_tensor.T

        # Reference calculation using high precision: M @ A.T = (A @ M.T).T
        reference = (reference_data @ v_np_64.T).T

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor (always float64)
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


def test_rmatmul_invalid_dims(sparse_tensor):
    """Test that rmatmul raises ValueError for tensors with invalid dimensions."""
    n, m = sparse_tensor.shape

    # 3D tensor (invalid for matmul)
    invalid_3d = torch.randn(3, 4, n, device=sparse_tensor.device)

    # Test __rmatmul__ with 3D tensor
    with pytest.raises(ValueError):
        _ = invalid_3d @ sparse_tensor

    # Also test with transposed matrix
    invalid_3d_t = torch.randn(3, 4, m, device=sparse_tensor.device)
    with pytest.raises(ValueError):
        _ = invalid_3d_t @ sparse_tensor.T
