"""Tests for custom DataLoader class."""

import numpy as np
import pytest
import torch

from rlaopt.data.dataloader import DataLoader, get_training_labels
from rlaopt.data.datasets import BatchedDataset, Dataset


class MockBatchedDataset(BatchedDataset):
    """Mock BatchedDataset for testing."""

    def __init__(self, num_samples, feature_dim, target_dim):
        super().__init__()
        self._num_samples = num_samples
        self._feature_dim = feature_dim
        self._target_dim = target_dim
        # Pre-generate deterministic data for consistency
        torch.manual_seed(42)
        self._data = torch.randn(num_samples, feature_dim)
        self._labels = (
            torch.randn(num_samples, target_dim)
            if target_dim > 1
            else torch.randn(num_samples)
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._data[idx], self._labels[idx]
        return self._data[idx], self._labels[idx]

    def __len__(self):
        return self._num_samples

    @property
    def feature_dimension(self):
        return self._feature_dim

    @property
    def target_dimension(self):
        return self._target_dim


@pytest.fixture
def in_memory_dataset():
    """Fixture providing an in-memory Dataset."""
    X = np.random.randn(30, 5).astype(np.float32)
    y = np.random.randn(30).astype(np.float32)
    return Dataset(X, y)


@pytest.fixture
def batched_dataset():
    """Fixture providing a BatchedDataset."""
    return MockBatchedDataset(40, 8, 3)


class TestDataLoaderInitialization:
    """Test DataLoader initialization and type checking."""

    def test_init_with_dataset(self, in_memory_dataset):
        """Test DataLoader initialization with an in-memory Dataset."""
        loader = DataLoader(in_memory_dataset, batch_size=8)
        assert loader.batch_size == 8
        assert isinstance(loader.dataset, Dataset)

    def test_init_with_batched_dataset(self, batched_dataset):
        """Test DataLoader initialization with a BatchedDataset."""
        loader = DataLoader(batched_dataset, batch_size=10)
        assert loader.batch_size == 10
        assert isinstance(loader.dataset, BatchedDataset)

    def test_invalid_dataset_type_raises(self):
        """Test that invalid dataset types raise TypeError."""
        invalid_data = [1, 2, 3, 4, 5]
        with pytest.raises(
            TypeError, match="Dataset must be of type Dataset or BatchedDataset"
        ):
            DataLoader(invalid_data)

    def test_invalid_torch_dataset_raises(self):
        """Test that standard PyTorch datasets raise TypeError."""
        torch_dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 5), torch.randn(10)
        )
        with pytest.raises(
            TypeError, match="Dataset must be of type Dataset or BatchedDataset"
        ):
            DataLoader(torch_dataset)


class TestDataLoaderYProperty:
    """Test the y property for retrieving training labels."""

    def test_y_property_in_memory_dataset(self, in_memory_dataset):
        """Test that y property returns all labels from in-memory dataset."""
        loader = DataLoader(in_memory_dataset, batch_size=5)
        labels = loader.y
        assert isinstance(labels, torch.Tensor)
        assert labels.shape[0] == 30
        assert torch.allclose(labels, in_memory_dataset.y)

    def test_y_property_batched_dataset(self, batched_dataset):
        """Test that y property returns all labels from batched dataset."""
        loader = DataLoader(batched_dataset, batch_size=10)
        labels = loader.y
        assert isinstance(labels, torch.Tensor)
        assert labels.shape == (40, 3)

    def test_y_property_batched_dataset_consistency(self, batched_dataset):
        """Test that y property returns consistent results across multiple calls."""
        loader = DataLoader(batched_dataset, batch_size=8, shuffle=False)
        labels1 = loader.y
        labels2 = loader.y
        assert torch.allclose(labels1, labels2)

    def test_y_property_1d_labels(self):
        """Test that y property handles 1D labels correctly."""
        X = np.random.randn(25, 4).astype(np.float32)
        y = np.random.randn(25).astype(np.float32)
        dataset = Dataset(X, y)
        loader = DataLoader(dataset, batch_size=5)
        labels = loader.y
        assert labels.ndim == 1
        assert labels.shape[0] == 25

    def test_y_property_multidimensional_labels(self):
        """Test that y property handles multidimensional labels correctly."""
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randn(20, 3, 2).astype(np.float32)
        dataset = Dataset(X, y)
        loader = DataLoader(dataset, batch_size=4)
        labels = loader.y
        assert labels.shape == (20, 3, 2)


class TestGetTrainingLabelsFunction:
    """Test the get_training_labels helper function."""

    def test_get_labels_in_memory(self, in_memory_dataset):
        """Test get_training_labels with in-memory dataset."""
        loader = DataLoader(in_memory_dataset, batch_size=6)
        labels = get_training_labels(loader, in_memory=True)
        assert torch.allclose(labels, in_memory_dataset.y)

    def test_get_labels_batched(self, batched_dataset):
        """Test get_training_labels with batched dataset."""
        loader = DataLoader(batched_dataset, batch_size=15, shuffle=False)
        labels = get_training_labels(loader, in_memory=False)
        assert labels.shape == (40, 3)
        assert isinstance(labels, torch.Tensor)

    def test_get_labels_batched_concatenation(self):
        """Test that get_training_labels correctly concatenates labels across batches."""
        dataset = MockBatchedDataset(17, 5, 1)  # Non-divisible by batch_size
        loader = DataLoader(dataset, batch_size=5, shuffle=False)
        labels = get_training_labels(loader, in_memory=False)
        assert labels.shape[0] == 17


class TestDataLoaderParameters:
    """Test DataLoader with various PyTorch parameters."""

    def test_shuffle_parameter(self, in_memory_dataset):
        """Test DataLoader initialization with shuffle parameter."""
        loader = DataLoader(in_memory_dataset, batch_size=8, shuffle=True)
        assert loader.batch_size == 8

    def test_drop_last_parameter(self, in_memory_dataset):
        """Test DataLoader with drop_last parameter drops incomplete batches."""
        loader = DataLoader(in_memory_dataset, batch_size=8, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3  # 30 // 8 = 3

    def test_num_workers_parameter(self, in_memory_dataset):
        """Test DataLoader initialization with num_workers parameter."""
        loader = DataLoader(in_memory_dataset, batch_size=5, num_workers=0)
        assert loader.num_workers == 0

    def test_pin_memory_parameter(self, in_memory_dataset):
        """Test DataLoader initialization with pin_memory parameter."""
        loader = DataLoader(in_memory_dataset, batch_size=5, pin_memory=False)
        assert loader.pin_memory is False


class TestDataLoaderFetching:
    """Test DataLoader data fetching behavior.

    Tests the iterator and get_batch return correct data.
    """

    def test_iteration_yields_correct_batches(self, in_memory_dataset):
        """Test that iterating over DataLoader yields correctly sized batches."""
        loader = DataLoader(in_memory_dataset, batch_size=10, shuffle=False)
        batch_count = 0
        for X_batch, y_batch in loader:
            assert X_batch.shape[0] <= 10
            assert y_batch.shape[0] <= 10
            batch_count += 1
        assert batch_count == 3  # 30 samples / 10 batch_size

    def test_get_batch_in_memory(self, in_memory_dataset):
        """Test get_batch method returns correct data from in-memory dataset."""
        loader = DataLoader(in_memory_dataset, batch_size=10, shuffle=False)
        X_batch, y_batch = loader.get_batch()
        X_true, y_true = in_memory_dataset[0:10]

        assert X_batch.shape[0] <= 10
        assert y_batch.shape[0] <= 10

        assert torch.allclose(X_batch, X_true)
        assert torch.allclose(y_batch, y_true)

    def test_iteration_with_batched_dataset(self, batched_dataset):
        """Test that iterating over DataLoader works with batched datasets."""
        loader = DataLoader(batched_dataset, batch_size=8, shuffle=False)
        batch_count = 0
        for X_batch, y_batch in loader:
            assert X_batch.shape == (8, 8)
            assert y_batch.shape == (8, 3)
            batch_count += 1
        assert batch_count == 5  # 40 samples / 8 batch_size

    def test_get_batch_with_batched_dataset(self, batched_dataset):
        """Test get_batch method returns correct data from batched dataset."""
        loader = DataLoader(batched_dataset, batch_size=10, shuffle=False)
        X_batch, y_batch = loader.get_batch()
        X_true, y_true = batched_dataset[0:10]

        assert X_batch.shape[0] <= 10
        assert y_batch.shape[0] <= 10

        assert torch.allclose(X_batch, X_true)
        assert torch.allclose(y_batch, y_true)
