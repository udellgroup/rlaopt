import pytest
import numpy as np
import scipy.sparse as sp
import torch

from rlaopt.sparse import SparseCSRTensor


# Fixture for test data preparation
@pytest.fixture(scope="module")
def sparse_data():
    """Generate a random sparse matrix for testing."""
    n = 100
    d = 100
    density = 0.05
    return sp.random_array((n, d), density=density, format="csr", dtype=np.float32)


# Fixture to create the sparse tensor for testing
@pytest.fixture(scope="module")
def sparse_tensor(sparse_data):
    """Create a SparseCSRTensor from the sparse data."""
    return SparseCSRTensor(sparse_data, device="cpu")


# Test cases
def test_shape(sparse_tensor, sparse_data):
    """Test the shape of the sparse tensor."""
    assert sparse_tensor.shape == sparse_data.shape


def test_ndim(sparse_tensor, sparse_data):
    """Test the number of dimensions of the sparse tensor."""
    assert sparse_tensor.ndim == sparse_data.ndim


def test_dtype(sparse_tensor):
    """Test the data type of the sparse tensor."""
    assert sparse_tensor.dtype == torch.float32


def test_movement(sparse_tensor):
    """Test the movement of sparse tensors between CPU and GPU."""
    if torch.cuda.is_available():
        sparse_tensor_moved = sparse_tensor.to(device="cuda:0")
        assert sparse_tensor_moved.device == torch.device("cuda:0")
        sparse_tensor_moved = sparse_tensor_moved.cpu()
        assert sparse_tensor_moved.device == torch.device("cpu")
        sparse_tensor_moved = sparse_tensor_moved.cuda()
        assert sparse_tensor_moved.device == torch.device("cuda:0")
    else:
        # If CUDA is not available, ensure the tensor remains on CPU
        sparse_tensor_moved = sparse_tensor.cuda()
        assert sparse_tensor_moved.device == torch.device("cpu")


# Parameterized test for scipy conversion
@pytest.mark.parametrize("transposed", [False, True])
def test_scipy_conversion(sparse_tensor, sparse_data, transposed):
    """Test conversion to and from scipy sparse format, with optional transposition."""
    if transposed:
        tensor_to_test = sparse_tensor.T
        data_to_compare = sparse_data.T
    else:
        tensor_to_test = sparse_tensor
        data_to_compare = sparse_data

    sparse_tensor_sp = tensor_to_test.scipy()
    assert sparse_tensor_sp.shape == data_to_compare.shape
    assert (sparse_tensor_sp.indptr == data_to_compare.indptr).all()
    assert (sparse_tensor_sp.indices == data_to_compare.indices).all()
    assert (sparse_tensor_sp.data == data_to_compare.data).all()


# Parameterized test for row slicing
@pytest.mark.parametrize(
    "slice_obj,get_numpy_equivalent",
    [
        # Basic slices
        (slice(0, 10), lambda s: s),
        (slice(None, 10), lambda s: s),
        (slice(90, None), lambda s: s),
        (slice(None, None, -1), lambda s: s),
        # Single indices
        (0, lambda s: s),
        (-1, lambda s: s),
        # List of indices
        ([3, 1, 2, 0], lambda s: s),
        # Tensor of indices
        (torch.tensor([3, 1, 2, 0]), lambda s: s.cpu().numpy()),
    ],
    ids=[
        "slice_0_10",
        "slice_start_10",
        "slice_90_end",
        "slice_reversed",
        "single_index_0",
        "single_index_neg1",
        "list_indices",
        "tensor_indices",
    ],
)
def test_row_slicing(sparse_tensor, sparse_data, slice_obj, get_numpy_equivalent):
    """Test row slicing of the sparse tensor."""
    # Handle callable slice_obj (for tensor creation)
    if callable(slice_obj):
        s = slice_obj()
    else:
        s = slice_obj

    # Get numpy equivalent for comparison
    s_compatible = get_numpy_equivalent(s)

    # Check if slicing works correctly
    assert (
        sparse_tensor[s].scipy().todense() == sparse_data[s_compatible].todense()
    ).all()


# Parameterized test for invalid row slicing
@pytest.mark.parametrize(
    "invalid_slice,expected_error",
    [
        # Test invalid matrix (CSC layout)
        (lambda tensor: tensor.T[0], ValueError),
        # Test invalid index (out of bounds)
        (lambda tensor: tensor[-101], IndexError),
        # Test invalid index (non-integer)
        (lambda tensor: tensor["invalid_index"], TypeError),
    ],
    ids=["csc_slice", "out_of_bounds", "non_integer"],
)
def test_invalid_row_slicing(sparse_tensor, invalid_slice, expected_error):
    """Test invalid row slicing of the sparse tensor."""
    with pytest.raises(expected_error):
        invalid_slice(sparse_tensor)
