"""
Tests for the datasets module.

This module tests all functionality of Dataset and BatchedDataset classes,
including construction from various data types, device handling, dtype conversion,
and edge cases.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from rlaopt.datasets import Dataset, BatchedDataset


class TestDataset:
    """Test suite for the Dataset class."""

    @pytest.fixture
    def sample_data_numpy(self):
        """Create sample numpy data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        return X, y

    @pytest.fixture
    def sample_data_pandas(self):
        """Create sample pandas data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        y = pd.Series(np.random.randn(100), name="target")
        return X, y

    @pytest.fixture
    def sample_data_torch(self):
        """Create sample torch tensor data for testing."""
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y = torch.randn(100)
        return X, y

    @pytest.fixture
    def sample_dataframe_combined(self):
        """Create sample DataFrame with features and targets combined."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "x1": np.random.randn(100),
                "x2": np.random.randn(100),
                "x3": np.random.randn(100),
                "y": np.random.randn(100),
            }
        )
        return df

    # Construction tests

    def test_init_from_numpy(self, sample_data_numpy):
        """Test Dataset initialization from numpy arrays."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        assert data.num_samples == 100
        assert data.feature_dimension == 10
        assert data.target_dimension == 1
        assert isinstance(data.X, torch.Tensor)
        assert isinstance(data.y, torch.Tensor)

    def test_init_from_pandas(self, sample_data_pandas):
        """Test Dataset initialization from pandas DataFrames."""
        X, y = sample_data_pandas
        data = Dataset(X, y)

        assert data.num_samples == 100
        assert data.feature_dimension == 10
        assert data.target_dimension == 1
        assert isinstance(data.X, torch.Tensor)
        assert isinstance(data.y, torch.Tensor)

    def test_init_from_torch(self, sample_data_torch):
        """Test Dataset initialization from torch tensors."""
        X, y = sample_data_torch
        data = Dataset(X, y)

        assert data.num_samples == 100
        assert data.feature_dimension == 10
        assert data.target_dimension == 1

    def test_from_numpy_classmethod(self, sample_data_numpy):
        """Test from_numpy classmethod."""
        X, y = sample_data_numpy
        data = Dataset.from_numpy(X, y)

        assert data.num_samples == 100
        assert data.feature_dimension == 10
        assert isinstance(data.X, torch.Tensor)

    def test_from_pandas_classmethod(self, sample_data_pandas):
        """Test from_pandas classmethod."""
        X, y = sample_data_pandas
        data = Dataset.from_pandas(X, y)

        assert data.num_samples == 100
        assert data.feature_dimension == 10
        assert isinstance(data.X, torch.Tensor)

    def test_from_dataframe_with_target_string(self, sample_dataframe_combined):
        """Test from_dataframe with target column as string."""
        df = sample_dataframe_combined
        data = Dataset.from_dataframe(df, target_cols="y")

        assert data.num_samples == 100
        assert data.feature_dimension == 3
        assert data.target_dimension == 1
        assert torch.allclose(data.y, torch.from_numpy(df["y"].values).float())

    def test_from_dataframe_with_target_list(self, sample_dataframe_combined):
        """Test from_dataframe with target column as list."""
        df = sample_dataframe_combined
        data = Dataset.from_dataframe(df, target_cols=["y"])

        assert data.num_samples == 100
        assert data.feature_dimension == 3
        assert data.target_dimension == 1

    def test_from_dataframe_with_feature_cols(self, sample_dataframe_combined):
        """Test from_dataframe with explicit feature columns."""
        df = sample_dataframe_combined
        data = Dataset.from_dataframe(df, target_cols="y", feature_cols=["x1", "x2"])

        assert data.num_samples == 100
        assert data.feature_dimension == 2
        assert data.target_dimension == 1

    def test_from_dataframe_multi_target(self):
        """Test from_dataframe with multiple target columns."""
        df = pd.DataFrame(
            {
                "x1": np.random.randn(50),
                "x2": np.random.randn(50),
                "y1": np.random.randn(50),
                "y2": np.random.randn(50),
                "y3": np.random.randn(50),
            }
        )
        data = Dataset.from_dataframe(df, target_cols=["y1", "y2", "y3"])

        assert data.num_samples == 50
        assert data.feature_dimension == 2
        assert data.target_dimension == (3,)

    # Multi-dimensional target tests

    def test_2d_target(self):
        """Test Dataset with 2D target (multi-target regression)."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 5)
        data = Dataset(X, y)

        assert data.target_dimension == (5,)
        assert data.y.shape == (100, 5)

    def test_3d_target(self):
        """Test Dataset with 3D target."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 3, 28, 28)
        data = Dataset(X, y)

        assert data.target_dimension == (3, 28, 28)
        assert data.y.shape == (100, 3, 28, 28)

    # Device handling tests

    def test_device_cpu(self, sample_data_numpy):
        """Test device placement on CPU."""
        X, y = sample_data_numpy
        data = Dataset(X, y, device="cpu")

        assert data.device == torch.device("cpu")
        assert data.X.device == torch.device("cpu")
        assert data.y.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, sample_data_numpy):
        """Test device placement on CUDA."""
        X, y = sample_data_numpy
        data = Dataset(X, y, device="cuda")

        assert data.device.type == "cuda"
        assert data.X.device.type == "cuda"
        assert data.y.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device(self, sample_data_numpy):
        """Test moving dataset to different device."""
        X, y = sample_data_numpy
        data = Dataset(X, y, device="cpu")
        data_cuda = data.to("cuda")

        assert data.device == torch.device("cpu")
        assert data_cuda.device.type == "cuda"
        # Original should be unchanged
        assert data.X.device == torch.device("cpu")

    # Dtype tests

    def test_dtype_default(self, sample_data_numpy):
        """Test default dtype is float32."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        assert data.dtype == torch.float32
        assert data.X.dtype == torch.float32
        assert data.y.dtype == torch.float32

    def test_dtype_float64(self, sample_data_numpy):
        """Test dtype conversion to float64."""
        X, y = sample_data_numpy
        data = Dataset(X, y, dtype=torch.float64)

        assert data.dtype == torch.float64
        assert data.X.dtype == torch.float64
        assert data.y.dtype == torch.float64

    def test_dtype_conversion_from_torch(self):
        """Test dtype conversion when input is already torch tensor."""
        X = torch.randn(100, 10, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64)
        data = Dataset(X, y, dtype=torch.float32)

        assert data.dtype == torch.float32
        assert data.X.dtype == torch.float32
        assert data.y.dtype == torch.float32

    def test_dtype_in_classmethods(self, sample_data_numpy):
        """Test dtype parameter in classmethods."""
        X, y = sample_data_numpy
        data = Dataset.from_numpy(X, y, dtype=torch.float64)

        assert data.dtype == torch.float64

    # Validation tests

    def test_invalid_X_dimensions(self):
        """Test that 1D X raises ValueError."""
        X = np.random.randn(100)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Dataset(X, y)

    def test_invalid_X_dimensions_3d(self):
        """Test that 3D X raises ValueError."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Dataset(X, y)

    def test_mismatched_sample_sizes(self):
        """Test that mismatched X and y sizes raise ValueError."""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="must have the same number of samples"):
            Dataset(X, y)

    # Indexing and iteration tests

    def test_getitem(self, sample_data_numpy):
        """Test __getitem__ access."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        sample_X, sample_y = data[0]
        assert sample_X.shape == (10,)
        assert sample_y.shape == ()

    def test_len(self, sample_data_numpy):
        """Test __len__."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        assert len(data) == 100

    def test_iteration(self, sample_data_numpy):
        """Test that dataset is iterable."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        count = 0
        for X_batch, y_batch in data:
            count += 1
            assert X_batch.shape == (10,)

        assert count == 100

    # Property tests

    def test_X_property(self, sample_data_numpy):
        """Test X property returns correct tensor."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        assert data.X.shape == (100, 10)
        assert torch.allclose(data.X, torch.from_numpy(X))

    def test_y_property(self, sample_data_numpy):
        """Test y property returns correct tensor."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        assert data.y.shape == (100,)
        assert torch.allclose(data.y, torch.from_numpy(y))

    def test_repr(self, sample_data_numpy):
        """Test __repr__ contains relevant information."""
        X, y = sample_data_numpy
        data = Dataset(X, y)

        repr_str = repr(data)
        assert "Dataset" in repr_str
        assert "num_samples=100" in repr_str
        assert "feature_dimension=10" in repr_str
        assert "target_dimension=1" in repr_str
        assert "dtype=torch.float32" in repr_str
        assert "device=" in repr_str

    # Edge cases

    def test_single_sample(self):
        """Test Dataset with single sample."""
        X = np.random.randn(1, 10)
        y = np.random.randn(1)
        data = Dataset(X, y)

        assert data.num_samples == 1
        assert len(data) == 1

    def test_large_feature_dimension(self):
        """Test Dataset with large feature dimension."""
        X = np.random.randn(10, 1000)
        y = np.random.randn(10)
        data = Dataset(X, y)

        assert data.feature_dimension == 1000

    def test_pandas_series_single_column(self):
        """Test with pandas Series for X (should work after conversion)."""
        # Note: Series will be converted to 1D array, which should fail
        X = pd.Series(np.random.randn(100))
        y = pd.Series(np.random.randn(100))

        with pytest.raises(ValueError, match="must be 2-dimensional"):
            Dataset(X, y)

    def test_pandas_dataframe_single_column(self):
        """Test with single-column DataFrame for X."""
        X = pd.DataFrame({"x1": np.random.randn(100)})
        y = pd.Series(np.random.randn(100))
        data = Dataset(X, y)

        assert data.feature_dimension == 1
        assert data.num_samples == 100


class ConcreteBatchedDataset(BatchedDataset):
    """Concrete implementation of BatchedDataset for testing."""

    def __init__(self, X, y):
        """Initialize with numpy arrays stored on disk simulation."""
        self.X_data = X
        self.y_data = y

    def __getitem__(self, idx):
        """Get item by index."""
        X_item = self.X_data[idx]
        y_item = self.y_data[idx]

        # Convert X
        X_tensor = torch.from_numpy(X_item).float()

        # Convert y - handle both array and scalar cases
        if np.ndim(y_item) == 0:  # scalar
            y_tensor = torch.tensor(y_item).float()
        else:  # array
            y_tensor = torch.from_numpy(y_item).float()

        return X_tensor, y_tensor

    def __len__(self):
        """Return length."""
        return len(self.X_data)

    @property
    def feature_dimension(self):
        """Return feature dimension."""
        return self.X_data.shape[1]

    @property
    def target_dimension(self):
        """Return target dimension."""
        if self.y_data.ndim == 1:
            return 1
        return self.y_data.shape[1:]


class TestBatchedDataset:
    """Test suite for the BatchedDataset abstract class."""

    @pytest.fixture
    def sample_batched_data(self):
        """Create sample data for BatchedDataset testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        return X, y

    def test_concrete_implementation(self, sample_batched_data):
        """Test that concrete implementation works."""
        X, y = sample_batched_data
        data = ConcreteBatchedDataset(X, y)

        assert data.num_samples == 100
        assert data.feature_dimension == 10
        assert data.target_dimension == 1

    def test_getitem(self, sample_batched_data):
        """Test __getitem__ on concrete implementation."""
        X, y = sample_batched_data
        data = ConcreteBatchedDataset(X, y)

        X_sample, y_sample = data[0]
        assert X_sample.shape == (10,)
        assert y_sample.shape == ()
        assert isinstance(X_sample, torch.Tensor)
        assert isinstance(y_sample, torch.Tensor)

    def test_len(self, sample_batched_data):
        """Test __len__ on concrete implementation."""
        X, y = sample_batched_data
        data = ConcreteBatchedDataset(X, y)

        assert len(data) == 100
        assert data.num_samples == 100

    def test_iteration(self, sample_batched_data):
        """Test iteration over BatchedDataset."""
        X, y = sample_batched_data
        data = ConcreteBatchedDataset(X, y)

        count = 0
        for X_batch, y_batch in data:
            count += 1
            assert isinstance(X_batch, torch.Tensor)
            assert isinstance(y_batch, torch.Tensor)

        assert count == 100

    def test_multi_dimensional_target(self):
        """Test BatchedDataset with multi-dimensional target."""
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 3, 5)
        data = ConcreteBatchedDataset(X, y)

        assert data.target_dimension == (3, 5)

    def test_cannot_instantiate_abstract(self):
        """Test that BatchedDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BatchedDataset()

    def test_dataloader_compatibility(self, sample_batched_data):
        """Test that BatchedDataset works with torch DataLoader."""
        X, y = sample_batched_data
        data = ConcreteBatchedDataset(X, y)

        dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)

        batch_count = 0
        for X_batch, y_batch in dataloader:
            batch_count += 1
            assert X_batch.shape == (10, 10)
            assert y_batch.shape == (10,)

        assert batch_count == 10


class TestIntegration:
    """Integration tests for datasets module."""

    def test_dataset_with_dataloader(self):
        """Test Dataset works seamlessly with PyTorch DataLoader."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 3)
        data = Dataset(X, y)

        dataloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

        for X_batch, y_batch in dataloader:
            assert X_batch.shape[0] <= 16
            assert X_batch.shape[1] == 10
            assert y_batch.shape[1] == 3
            break  # Just test one batch

    def test_mixed_dtypes_conversion(self):
        """Test that mixed dtypes are properly converted."""
        X = np.random.randn(50, 5).astype(np.float64)
        y = np.random.randn(50).astype(np.float32)
        data = Dataset(X, y, dtype=torch.float32)

        assert data.X.dtype == torch.float32
        assert data.y.dtype == torch.float32

    def test_pandas_to_torch_pipeline(self):
        """Test complete pipeline from pandas to torch."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40],
                "income": [50000, 60000, 75000, 80000],
                "score": [0.5, 0.6, 0.7, 0.8],
            }
        )

        data = Dataset.from_dataframe(df, target_cols="score", dtype=torch.float32)

        assert data.num_samples == 4
        assert data.feature_dimension == 2
        assert data.dtype == torch.float32

        # Test that data is correctly converted
        assert torch.allclose(data.y, torch.tensor([0.5, 0.6, 0.7, 0.8]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
