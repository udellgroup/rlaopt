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
    return SparseCSRTensor(data=sparse_data, device=device)


# Test configuration for all matrix multiplication tests
@pytest.mark.parametrize(
    "op_type,shape_type",
    [
        # Format: (operation_type, input_shape_type)
        ("matmul", "1d"),  # sparse @ vector
        ("matmul", "2d"),  # sparse @ matrix
        ("matmul_t", "1d"),  # sparse.T @ vector
        ("matmul_t", "2d"),  # sparse.T @ matrix
        ("rmatmul", "1d"),  # vector @ sparse
        ("rmatmul", "2d"),  # matrix @ sparse
        ("rmatmul_t", "1d"),  # vector @ sparse.T
        ("rmatmul_t", "2d"),  # matrix @ sparse.T
    ],
    ids=[
        "sparse@vector",
        "sparse@matrix",
        "sparse.T@vector",
        "sparse.T@matrix",
        "vector@sparse",
        "matrix@sparse",
        "vector@sparse.T",
        "matrix@sparse.T",
    ],
)
def test_all_matmul_operations(
    op_type, shape_type, sparse_tensor, reference_data, device, tol
):
    """Universal test for all matrix multiplication operations."""
    n, m = sparse_tensor.shape
    batch_size = 5

    # Determine dimensions based on operation and shape type
    if op_type in ["matmul", "rmatmul_t"]:
        dim1 = m
        sparse_op = sparse_tensor if op_type == "matmul" else sparse_tensor.T
    else:  # matmul_t or rmatmul
        dim1 = n
        sparse_op = sparse_tensor.T if op_type == "matmul_t" else sparse_tensor

    # Create input tensor based on shape type parameter
    if shape_type == "1d":
        # 1D vector case
        v = torch.randn(dim1, device=device)
        v_np_64 = v.cpu().numpy().astype(np.float64)

        # Calculate using the appropriate operation
        if op_type.startswith("matmul"):
            result = sparse_op @ v
            # Reference calculation
            ref_mat = reference_data if op_type == "matmul" else reference_data.T
            reference = ref_mat @ v_np_64
        else:  # rmatmul
            result = v @ sparse_op
            # Reference calculation
            ref_mat = reference_data.T if op_type == "rmatmul" else reference_data
            reference = ref_mat @ v_np_64
    else:  # 2d
        # 2D matrix case
        if op_type.startswith("matmul"):
            # For matmul, second dimension can be arbitrary
            v = torch.randn(dim1, batch_size, device=device)
            v_np_64 = v.cpu().numpy().astype(np.float64)

            # Calculate using the appropriate operation
            result = sparse_op @ v

            # Reference calculation
            ref_mat = reference_data if op_type == "matmul" else reference_data.T
            reference = ref_mat @ v_np_64
        else:  # rmatmul
            # For rmatmul, first dimension can be arbitrary
            batch_size = 5
            v = torch.randn(batch_size, dim1, device=device)
            v_np_64 = v.cpu().numpy().astype(np.float64)

            # Calculate using rmatmul
            result = v @ sparse_op

            # Reference calculation
            if op_type == "rmatmul":
                reference = (reference_data.T @ v_np_64.T).T
            else:  # rmatmul_t
                reference = (reference_data @ v_np_64.T).T

    # Convert result to float64 for comparison
    result_64 = result.to(torch.float64)

    # Convert reference to torch tensor
    reference_tensor = torch.tensor(reference, device=device, dtype=torch.float64)

    # Verify results
    assert torch.allclose(result_64, reference_tensor, **tol)


# Simplified test for invalid dimensions
@pytest.mark.parametrize(
    "op_type",
    ["matmul", "matmul_t", "rmatmul", "rmatmul_t"],
    ids=["sparse@x", "sparse.T@x", "x@sparse", "x@sparse.T"],
)
def test_invalid_dims(op_type, sparse_tensor):
    """Test that operations raise ValueError for tensors with invalid dimensions."""
    n, m = sparse_tensor.shape

    # Determine dimensions based on operation
    if op_type in ["matmul", "rmatmul_t"]:
        dim = m
        sparse_op = sparse_tensor
    else:  # matmul_t or rmatmul
        dim = n
        sparse_op = sparse_tensor.T if op_type == "matmul_t" else sparse_tensor

    # Create 3D tensor (invalid for matmul)
    invalid_3d_shape = (dim, 3, 4) if op_type.startswith("matmul") else (3, 4, dim)
    invalid_3d = torch.randn(invalid_3d_shape, device=sparse_tensor.device)

    # Test operation with 3D tensor
    with pytest.raises(ValueError):
        if op_type.startswith("matmul"):
            _ = sparse_op @ invalid_3d
        else:  # rmatmul
            _ = invalid_3d @ sparse_op
