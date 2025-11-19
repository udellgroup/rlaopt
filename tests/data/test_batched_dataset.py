"""Tests for BatchedDataset abstract class."""

import pytest
import torch

from rlaopt.data.datasets import BatchedDataset


class ConcreteBatchedDataset(BatchedDataset):
    """Concrete implementation for testing BatchedDataset interface."""

    def __init__(self, num_samples, feature_dim, target_dim):
        super().__init__()
        self._num_samples = num_samples
        self._feature_dim = feature_dim
        self._target_dim = target_dim
        # Simulate large dataset metadata
        self.data_shape = (num_samples, feature_dim)
        self.target_shape = (num_samples,) + (
            target_dim if isinstance(target_dim, tuple) else (target_dim,)
        )

    def __getitem__(self, idx):
        """Generate synthetic data for the given index."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._num_samples)
            length = len(range(start, stop, step))
            X = torch.randn(length, self._feature_dim)
            if isinstance(self._target_dim, tuple):
                y = torch.randn(length, *self._target_dim)
            else:
                y = torch.randn(length, self._target_dim)
        else:
            X = torch.randn(self._feature_dim)
            if isinstance(self._target_dim, tuple):
                y = torch.randn(*self._target_dim)
            else:
                y = torch.randn(self._target_dim)
        return X, y

    def __len__(self):
        return self._num_samples

    @property
    def feature_dimension(self):
        return self._feature_dim

    @property
    def target_dimension(self):
        return self._target_dim


class TestBatchedDatasetInterface:
    """Test BatchedDataset abstract interface requirements."""

    def test_cannot_instantiate_abstract_class(self):
        """BatchedDataset should not be instantiable directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BatchedDataset()

    def test_concrete_implementation_works(self):
        """Concrete implementation should work correctly."""
        dataset = ConcreteBatchedDataset(100, 10, 1)
        assert len(dataset) == 100
        assert dataset.num_samples == 100
        assert dataset.feature_dimension == 10
        assert dataset.target_dimension == 1


class TestBatchedDatasetGetitem:
    """Test __getitem__ implementation for batched access."""

    @pytest.fixture
    def dataset(self):
        """Create a concrete batched dataset fixture for testing.

        Returns:
            ConcreteBatchedDataset: A dataset with 50 samples, 8 features, and 3 target dimensions.
        """
        return ConcreteBatchedDataset(50, 8, 3)

    def test_single_sample_access(self, dataset):
        """Test that accessing a single sample returns correct tensor shapes."""
        X, y = dataset[0]
        assert X.shape == (8,)
        assert y.shape == (3,)

    def test_slice_access(self, dataset):
        """Test that slice indexing returns batched tensors with correct shapes."""
        X, y = dataset[0:10]
        assert X.shape == (10, 8)
        assert y.shape == (10, 3)

    def test_slice_with_step(self, dataset):
        """Test that slice indexing with step parameter works correctly."""
        X, y = dataset[0:20:2]
        assert X.shape == (10, 8)
        assert y.shape == (10, 3)

    def test_negative_indexing(self, dataset):
        """Test that negative indexing correctly accesses samples from the end."""
        X, y = dataset[-1]
        assert X.shape == (8,)
        assert y.shape == (3,)


class TestBatchedDatasetProperties:
    """Test properties of BatchedDataset implementations."""

    def test_num_samples_from_len(self):
        """Test that num_samples property matches len() for the dataset."""
        dataset = ConcreteBatchedDataset(75, 5, 2)
        assert dataset.num_samples == 75
        assert dataset.num_samples == len(dataset)

    def test_feature_dimension_scalar(self):
        """Test that feature_dimension property returns correct scalar value."""
        dataset = ConcreteBatchedDataset(100, 15, 1)
        assert dataset.feature_dimension == 15
        assert isinstance(dataset.feature_dimension, int)

    def test_target_dimension_scalar(self):
        """Test that target_dimension property returns correct scalar value."""
        dataset = ConcreteBatchedDataset(100, 10, 5)
        assert dataset.target_dimension == 5

    def test_target_dimension_tuple(self):
        """Test that target_dimension property handles multidimensional targets correctly."""
        dataset = ConcreteBatchedDataset(100, 10, (3, 28, 28))
        assert dataset.target_dimension == (3, 28, 28)
        assert isinstance(dataset.target_dimension, tuple)


class TestBatchedDatasetIteration:
    """Test iteration capabilities of BatchedDataset."""

    def test_iteration_with_dataloader(self):
        """Test that dataset works correctly with PyTorch DataLoader."""
        dataset = ConcreteBatchedDataset(20, 5, 2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        batch_count = 0
        for X_batch, y_batch in dataloader:
            assert X_batch.shape == (4, 5)
            assert y_batch.shape == (4, 2)
            batch_count += 1

        assert batch_count == 5  # 20 samples / 4 batch_size

    def test_manual_iteration(self):
        """Test that manual iteration over dataset indices works correctly."""
        dataset = ConcreteBatchedDataset(10, 3, 1)
        samples_collected = 0

        for i in range(len(dataset)):
            X, y = dataset[i]
            assert X.shape == (3,)
            assert y.shape == (1,)
            samples_collected += 1

        assert samples_collected == 10


class TestBatchedDatasetEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_dataset(self):
        """Test that dataset handles single-sample case correctly."""
        dataset = ConcreteBatchedDataset(1, 5, 1)
        assert len(dataset) == 1
        X, y = dataset[0]
        assert X.shape == (5,)

    def test_large_feature_dimension(self):
        """Test that dataset handles high-dimensional feature spaces correctly."""
        dataset = ConcreteBatchedDataset(10, 1000, 1)
        X, y = dataset[0]
        assert X.shape == (1000,)

    def test_multidimensional_targets(self):
        """Test that dataset handles multidimensional target shapes correctly."""
        dataset = ConcreteBatchedDataset(10, 5, (3, 64, 64))
        X, y = dataset[0]
        assert X.shape == (5,)
        assert y.shape == (3, 64, 64)
