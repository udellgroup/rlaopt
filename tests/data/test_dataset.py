"""Tests for Dataset class."""

import numpy as np
import pandas as pd
import pytest
import torch

from rlaopt.data.datasets import Dataset


@pytest.fixture
def sample_data():
    """Fixture providing sample numpy arrays."""
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    return X, y


@pytest.fixture
def classification_data():
    """Fixture providing classification data with integer labels."""
    X = np.random.randn(15, 4).astype(np.float32)
    y = np.array([0, 1, 2] * 5, dtype=np.int32)
    return X, y


class TestDatasetInitialization:
    """Test Dataset initialization with various input types."""

    def test_from_numpy(self, sample_data):
        """Test Dataset initialization from numpy arrays."""
        X, y = sample_data
        data = Dataset(X, y)
        assert data.num_samples == 20
        assert data.feature_dimension == 5
        assert data.target_dimension == 1
        assert isinstance(data.X, torch.Tensor)
        assert isinstance(data.y, torch.Tensor)

    def test_from_pandas(self, sample_data):
        """Test Dataset initialization from pandas DataFrame and Series."""
        X, y = sample_data
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        data = Dataset(X_df, y_series)
        assert data.num_samples == 20
        assert torch.allclose(data.X, torch.from_numpy(X))

    def test_from_tensors(self, sample_data):
        """Test Dataset initialization from PyTorch tensors."""
        X, y = sample_data
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        data = Dataset(X_t, y_t)
        assert data.num_samples == 20

    def test_multitarget(self, sample_data):
        """Test Dataset initialization with multidimensional targets."""
        X, _ = sample_data
        y_multi = np.random.randn(20, 3).astype(np.float32)
        data = Dataset(X, y_multi)
        assert data.target_dimension == (3,)

    def test_classification_preserves_int(self, classification_data):
        """Test that integer target labels are preserved for classification tasks."""
        X, y = classification_data
        data = Dataset(X, y)
        assert data.y.dtype == torch.long
        assert torch.all(data.y >= 0)


class TestDatasetValidation:
    """Test Dataset validation and error handling."""

    def test_1d_input_raises(self, sample_data):
        """Test that 1D feature input raises ValueError."""
        _, y = sample_data
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Dataset(y, y)

    def test_mismatched_samples_raises(self, sample_data):
        """Test that mismatched sample counts between X and y raise ValueError."""
        X, y = sample_data
        y_wrong = y[:10]
        with pytest.raises(ValueError, match="same number of samples"):
            Dataset(X, y_wrong)

    def test_3d_input_raises(self):
        """Test that 3D feature input raises ValueError."""
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Dataset(X, y)


class TestDatasetClassmethods:
    """Test Dataset factory methods."""

    def test_from_numpy_classmethod(self, sample_data):
        """Test Dataset.from_numpy() factory method with dtype specification."""
        X, y = sample_data
        data = Dataset.from_numpy(X, y, dtype=torch.float64)
        assert data.dtype == torch.float64

    def test_from_pandas_classmethod(self, sample_data):
        """Test Dataset.from_pandas() factory method."""
        X, y = sample_data
        data = Dataset.from_pandas(pd.DataFrame(X), pd.Series(y))
        assert data.num_samples == 20

    def test_from_dataframe_single_target(self):
        """Test Dataset.from_dataframe() with a single target column."""
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [7, 8, 9]})
        data = Dataset.from_dataframe(df, target_cols="y")
        assert data.feature_dimension == 2
        assert data.target_dimension == 1
        assert torch.allclose(data.y, torch.tensor([7, 8, 9]))

    def test_from_dataframe_multi_target(self):
        """Test Dataset.from_dataframe() with multiple target columns."""
        df = pd.DataFrame({"x": [1, 2], "y1": [3, 4], "y2": [5, 6]})
        data = Dataset.from_dataframe(df, target_cols=["y1", "y2"])
        assert data.feature_dimension == 1
        assert data.target_dimension == (2,)

    def test_from_dataframe_explicit_features(self):
        """Test Dataset.from_dataframe() with explicitly specified feature columns."""
        df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "x3": [5, 6], "y": [7, 8]})
        data = Dataset.from_dataframe(df, target_cols="y", feature_cols=["x1", "x2"])
        assert data.feature_dimension == 2


class TestDatasetDeviceManagement:
    """Test device placement and movement."""

    def test_device_cpu(self, sample_data):
        """Test Dataset initialization on CPU device."""
        X, y = sample_data
        data = Dataset(X, y, device="cpu")
        assert data.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, sample_data):
        """Test Dataset initialization on CUDA device."""
        X, y = sample_data
        data = Dataset(X, y, device="cuda")
        assert data.device.type == "cuda"

    def test_to_method(self, sample_data):
        """Test Dataset.to() method for device transfer."""
        X, y = sample_data
        data = Dataset(X, y, device="cpu")
        data_new = data.to("cpu")
        assert isinstance(data_new, Dataset)
        assert data_new.device == torch.device("cpu")


class TestDatasetProperties:
    """Test Dataset properties and methods."""

    def test_repr(self, sample_data):
        """Test Dataset string representation contains key information."""
        X, y = sample_data
        data = Dataset(X, y)
        repr_str = repr(data)
        assert "num_samples=20" in repr_str
        assert "feature_dimension=5" in repr_str
        assert "target_dimension=1" in repr_str

    def test_getitem(self, sample_data):
        """Test Dataset indexing returns correct sample shapes."""
        X, y = sample_data
        data = Dataset(X, y)
        X_sample, y_sample = data[0]
        assert X_sample.shape == (5,)
        assert y_sample.shape == ()

    def test_len(self, sample_data):
        """Test Dataset length matches number of samples."""
        X, y = sample_data
        data = Dataset(X, y)
        assert len(data) == 20

    def test_target_dimension_3d(self):
        """Test Dataset correctly handles 3D multidimensional targets."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 3, 28, 28)
        data = Dataset(X, y)
        assert data.target_dimension == (3, 28, 28)
