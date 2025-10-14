"""
Comprehensive test suite for the GLM core module.

Tests cover:
- compute_loss with different data sources
- compute_prediction with different data sources
- batch_loss and batch_predict
- _has_test_data helper
- Edge cases and error handling
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from rlaopt.datasets import Dataset, BatchedDataset
from rlaopt.dataloader import DataLoader
from rlaopt.atoms._glm.core import (
    compute_loss,
    compute_prediction,
    batch_loss,
    batch_predict,
    _has_test_data,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = torch.randn(100, 10)
    y = torch.randn(100)
    beta = torch.randn(10)
    return X, y, beta


@pytest.fixture
def small_batch_data():
    """Create small batch data for testing."""
    X_batch = torch.randn(32, 10)
    y_batch = torch.randn(32)
    beta = torch.randn(10)
    return X_batch, y_batch, beta


@pytest.fixture
def dataset_and_loader(sample_data):
    """Create Dataset and DataLoader."""
    X, y, _ = sample_data
    dataset = Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataset, dataloader


@pytest.fixture
def mock_loss_fn():
    """Create a mock loss function using torch.nn.MSELoss."""
    return torch.nn.MSELoss(reduction="mean")


# ============================================================================
# Tests for _has_test_data
# ============================================================================


class TestHasTestData:
    """Test suite for _has_test_data helper function."""

    def test_both_none_returns_false(self):
        """Test that both None returns False."""
        assert _has_test_data(None, None) is False

    def test_x_provided_y_none_raises_error(self):
        """Test that X without y raises ValueError."""
        X = torch.randn(10, 5)
        with pytest.raises(ValueError, match="Must provide y when X is specified"):
            _has_test_data(X, None)

    def test_both_provided_returns_true(self):
        """Test that both X and y returns True."""
        X = torch.randn(10, 5)
        y = torch.randn(10)
        assert _has_test_data(X, y) is True

    def test_y_provided_x_none_returns_false(self):
        """Test that only y provided returns False."""
        y = torch.randn(10)
        assert _has_test_data(None, y) is False


# ============================================================================
# Tests for batch_predict
# ============================================================================


class TestBatchPredict:
    """Test suite for batch_predict function."""

    def test_basic_prediction(self, small_batch_data):
        """Test basic batch prediction."""
        X_batch, y_batch, beta = small_batch_data
        pred, y_out = batch_predict(beta, X_batch, y_batch)

        # Check shapes
        assert pred.shape == (32,)
        assert y_out.shape == (32,)

        # Check prediction is correct (X @ beta)
        expected = X_batch @ beta
        assert torch.allclose(pred, expected)

        # Check y is passed through
        assert torch.allclose(y_out, y_batch)

    def test_prediction_without_y(self, small_batch_data):
        """Test batch prediction without y."""
        X_batch, _, beta = small_batch_data
        pred, y_out = batch_predict(beta, X_batch, y_batch=None)

        assert pred.shape == (32,)
        assert y_out is None

    def test_gradient_preservation(self, small_batch_data):
        """Test that gradients are preserved."""
        X_batch, y_batch, _ = small_batch_data
        beta = torch.randn(10, requires_grad=True)

        pred, _ = batch_predict(beta, X_batch, y_batch)
        loss = pred.sum()
        loss.backward()

        assert beta.grad is not None

    @patch("rlaopt.atoms._glm.core.move_to_source_device")
    def test_device_movement_with_y(self, mock_move, small_batch_data):
        """Test that device movement is called when y is provided."""
        X_batch, y_batch, beta = small_batch_data
        mock_move.return_value = (X_batch, y_batch)

        batch_predict(beta, X_batch, y_batch)

        mock_move.assert_called_once()
        args = mock_move.call_args[0]
        assert len(args[0]) == 2  # tuple of (X_batch, y_batch)

    @patch("rlaopt.atoms._glm.core.move_to_source_device")
    def test_device_movement_without_y(self, mock_move, small_batch_data):
        """Test that device movement is called when y is None."""
        X_batch, _, beta = small_batch_data
        mock_move.return_value = X_batch

        batch_predict(beta, X_batch, y_batch=None)

        mock_move.assert_called_once()


# ============================================================================
# Tests for batch_loss
# ============================================================================


class TestBatchLoss:
    """Test suite for batch_loss function."""

    def test_normalized_loss(self, small_batch_data, mock_loss_fn):
        """Test batch loss with normalization."""
        X_batch, y_batch, beta = small_batch_data

        loss = batch_loss(beta, X_batch, y_batch, mock_loss_fn, normalize=True)

        # Check it's a scalar
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_unnormalized_loss(self, small_batch_data, mock_loss_fn):
        """Test batch loss without normalization (sum reduction)."""
        X_batch, y_batch, beta = small_batch_data

        loss_unnorm = batch_loss(beta, X_batch, y_batch, mock_loss_fn, normalize=False)
        loss_norm = batch_loss(beta, X_batch, y_batch, mock_loss_fn, normalize=True)

        # Unnormalized should be larger (it's the sum)
        assert loss_unnorm > loss_norm
        assert loss_unnorm.shape == torch.Size([])

    def test_reduction_restored(self, small_batch_data):
        """Test that loss function reduction is restored after unnormalized loss."""
        X_batch, y_batch, beta = small_batch_data

        # Use actual torch loss function
        loss_fn = torch.nn.MSELoss(reduction="mean")

        batch_loss(beta, X_batch, y_batch, loss_fn, normalize=False)

        # Check reduction was restored to mean
        assert loss_fn.reduction == "mean"

    def test_gradient_flow(self, small_batch_data, mock_loss_fn):
        """Test that gradients flow through batch loss."""
        X_batch, y_batch, _ = small_batch_data
        beta = torch.randn(10, requires_grad=True)

        loss = batch_loss(beta, X_batch, y_batch, mock_loss_fn, normalize=True)
        loss.backward()

        assert beta.grad is not None


# ============================================================================
# Tests for compute_prediction
# ============================================================================


class TestComputePrediction:
    """Test suite for compute_prediction function."""

    def test_prediction_with_new_data(self, sample_data):
        """Test prediction on new inference data."""
        X, _, beta = sample_data
        X_test = torch.randn(20, 10)

        # When X is provided, dataloader can be anything (it's not used)
        pred = compute_prediction(beta, dataloader=None, X=X_test)

        assert pred.shape == (20,)
        expected = X_test @ beta
        assert torch.allclose(pred, expected)

    def test_prediction_with_dataset(self, dataset_and_loader, sample_data):
        """Test prediction using full in-memory Dataset."""
        dataset, dataloader = dataset_and_loader
        _, _, beta = sample_data

        pred = compute_prediction(beta, dataloader=dataloader)

        assert pred.shape == (100,)
        expected = dataset.X @ beta
        assert torch.allclose(pred, expected)

    def test_prediction_batched(self, sample_data):
        """Test prediction using batched approach with BatchedDataset."""
        X, y, beta = sample_data

        # Create a proper BatchedDataset subclass with required abstract methods
        class MockBatchedDataset(BatchedDataset):
            def __init__(self, X, y):
                self._X = X
                self._y = y

            def __len__(self):
                return len(self._X)

            def __getitem__(self, idx):
                return self._X[idx], self._y[idx]

            @property
            def feature_dimension(self):
                return self._X.shape[1]

            @property
            def target_dimension(self):
                return 1 if self._y.dim() == 1 else self._y.shape[1]

        batched_dataset = MockBatchedDataset(X, y)
        dataloader = DataLoader(batched_dataset, batch_size=32, shuffle=False)

        pred = compute_prediction(beta, dataloader=dataloader)

        assert pred.shape == (100,)
        # Verify the prediction is correct
        expected = X @ beta
        assert torch.allclose(pred, expected, rtol=1e-5)

    def test_prediction_raises_without_dataloader(self):
        """Test that prediction raises error without dataloader for batched data."""
        beta = torch.randn(10)

        # This should raise because dataloader is None and X is None
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'dataset'"
        ):
            compute_prediction(beta, dataloader=None, X=None)

    def test_gradient_preservation_inference(self, sample_data):
        """Test that gradients are preserved during inference."""
        X_test = torch.randn(20, 10)
        beta = torch.randn(10, requires_grad=True)

        pred = compute_prediction(beta, dataloader=None, X=X_test)
        loss = pred.sum()
        loss.backward()

        assert beta.grad is not None

    def test_gradient_preservation_training(self, dataset_and_loader):
        """Test that gradients are preserved when using training data."""
        _, dataloader = dataset_and_loader
        beta = torch.randn(10, requires_grad=True)

        pred = compute_prediction(beta, dataloader=dataloader)
        loss = pred.sum()
        loss.backward()

        assert beta.grad is not None

    @patch("rlaopt.atoms._glm.core.move_to_source_device")
    def test_device_movement(self, mock_move, sample_data):
        """Test that device movement is called for inference data."""
        X_test = torch.randn(20, 10)
        _, _, beta = sample_data
        mock_move.return_value = X_test

        compute_prediction(beta, dataloader=None, X=X_test)

        mock_move.assert_called_once_with(X_test, beta)


# ============================================================================
# Tests for compute_loss
# ============================================================================


class TestComputeLoss:
    """Test suite for compute_loss function."""

    def test_loss_with_test_data(self, sample_data, mock_loss_fn):
        """Test loss computation with test data provided."""
        X, y, beta = sample_data
        X_test = torch.randn(20, 10)
        y_test = torch.randn(20)

        loss = compute_loss(beta, mock_loss_fn, None, X=X_test, y=y_test)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_loss_with_dataset(self, dataset_and_loader, sample_data, mock_loss_fn):
        """Test loss computation with in-memory Dataset."""
        dataset, dataloader = dataset_and_loader
        _, _, beta = sample_data

        loss = compute_loss(beta, mock_loss_fn, dataloader)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_loss_with_batched_data(self, sample_data, mock_loss_fn):
        """Test loss computation with batched data."""
        X, y, beta = sample_data

        # Create a proper BatchedDataset subclass with required abstract methods
        class MockBatchedDataset(BatchedDataset):
            def __init__(self, X, y):
                self._X = X
                self._y = y

            def __len__(self):
                return len(self._X)

            def __getitem__(self, idx):
                return self._X[idx], self._y[idx]

            @property
            def feature_dimension(self):
                return self._X.shape[1]

            @property
            def target_dimension(self):
                return 1 if self._y.dim() == 1 else self._y.shape[1]

        batched_dataset = MockBatchedDataset(X, y)
        dataloader = DataLoader(batched_dataset, batch_size=32, shuffle=False)

        loss = compute_loss(beta, mock_loss_fn, dataloader)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_loss_raises_without_dataloader(self):
        """Test that loss raises error when dataloader is None."""
        beta = torch.randn(10)
        mock_loss_fn = torch.nn.MSELoss(reduction="mean")

        # This should raise AttributeError when trying to access dataloader.dataset
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'dataset'"
        ):
            compute_loss(beta, mock_loss_fn, dataloader=None)

    def test_loss_gradient_flow(self, dataset_and_loader, mock_loss_fn):
        """Test that gradients flow through loss computation."""
        _, dataloader = dataset_and_loader
        beta = torch.randn(10, requires_grad=True)

        loss = compute_loss(beta, mock_loss_fn, dataloader)
        loss.backward()

        assert beta.grad is not None

    def test_loss_batched_averaging(self, sample_data, mock_loss_fn):
        """Test that batched loss properly averages across batches."""
        X, y, beta = sample_data

        # Create dataset with Dataset type (fast path)
        dataset = Dataset(X, y)
        dataloader_full = DataLoader(dataset, batch_size=32, shuffle=False)

        # Compute loss with Dataset (full dataset at once)
        full_loss = compute_loss(beta, mock_loss_fn, dataloader_full)

        # Create BatchedDataset subclass (slow path - iterates batches)
        class MockBatchedDataset(BatchedDataset):
            def __init__(self, X, y):
                self._X = X
                self._y = y

            def __len__(self):
                return len(self._X)

            def __getitem__(self, idx):
                return self._X[idx], self._y[idx]

            @property
            def feature_dimension(self):
                return self._X.shape[1]

            @property
            def target_dimension(self):
                return 1 if self._y.dim() == 1 else self._y.shape[1]

        batched_dataset = MockBatchedDataset(X, y)
        dataloader_batched = DataLoader(batched_dataset, batch_size=32, shuffle=False)

        # Compute loss with batched approach
        batched_loss = compute_loss(beta, mock_loss_fn, dataloader_batched)

        # They should be approximately equal (within numerical precision)
        assert torch.allclose(full_loss, batched_loss, rtol=1e-5)

    def test_loss_with_parameter(self, dataset_and_loader, mock_loss_fn):
        """Test loss computation with torch.nn.Parameter."""
        _, dataloader = dataset_and_loader
        beta = torch.nn.Parameter(torch.randn(10))

        loss = compute_loss(beta, mock_loss_fn, dataloader)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_training_workflow(self, dataset_and_loader, mock_loss_fn):
        """Test a complete training step workflow."""
        _, dataloader = dataset_and_loader
        beta = torch.nn.Parameter(torch.randn(10))
        optimizer = torch.optim.SGD([beta], lr=0.01)

        # Training step
        optimizer.zero_grad()
        loss = compute_loss(beta, mock_loss_fn, dataloader)
        loss.backward()
        optimizer.step()

        # Check everything worked
        assert loss.item() >= 0
        assert beta.grad is not None

    def test_train_test_split(self, sample_data, mock_loss_fn):
        """Test training on one dataset and evaluating on another."""
        X, y, _ = sample_data

        # Split data
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]

        # Create training dataloader
        train_dataset = Dataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32)

        # Initialize model
        beta = torch.nn.Parameter(torch.randn(10))

        # Compute training loss
        train_loss = compute_loss(beta, mock_loss_fn, train_loader)

        # Compute test loss
        test_loss = compute_loss(beta, mock_loss_fn, train_loader, X=X_test, y=y_test)

        assert train_loss.item() >= 0
        assert test_loss.item() >= 0

    def test_different_batch_sizes(self, sample_data, mock_loss_fn):
        """Test that different batch sizes give same result."""
        X, y, beta = sample_data

        dataset = Dataset(X, y)
        loader_32 = DataLoader(dataset, batch_size=32, shuffle=False)
        loader_16 = DataLoader(dataset, batch_size=16, shuffle=False)

        loss_32 = compute_loss(beta, mock_loss_fn, loader_32)
        loss_16 = compute_loss(beta, mock_loss_fn, loader_16)

        # Should be approximately equal
        assert torch.allclose(loss_32, loss_16, rtol=1e-5)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch(self, mock_loss_fn):
        """Test handling of empty batch."""
        beta = torch.randn(10)
        X_batch = torch.randn(0, 10)
        y_batch = torch.randn(0)

        pred, _ = batch_predict(beta, X_batch, y_batch)
        assert pred.shape == (0,)

    def test_single_sample(self, mock_loss_fn):
        """Test with single sample."""
        beta = torch.randn(10)
        X = torch.randn(1, 10)
        y = torch.randn(1)

        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1)

        loss = compute_loss(beta, mock_loss_fn, dataloader)
        assert loss.shape == torch.Size([])

    def test_large_features(self, mock_loss_fn):
        """Test with large feature dimension."""
        X = torch.randn(50, 1000)
        y = torch.randn(50)
        beta = torch.randn(1000)

        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)

        loss = compute_loss(beta, mock_loss_fn, dataloader)
        assert loss.shape == torch.Size([])

    def test_prediction_shape_mismatch_caught(self, mock_loss_fn):
        """Test that shape mismatches are caught."""
        beta = torch.randn(10)
        X = torch.randn(20, 5)  # Wrong feature dimension
        y = torch.randn(20)

        with pytest.raises(RuntimeError):  # Matrix multiplication will fail
            pred = compute_prediction(beta, dataloader=None, X=X)

    def test_loss_with_nan_values(self, dataset_and_loader):
        """Test behavior with NaN values in loss function."""
        _, dataloader = dataset_and_loader
        beta = torch.randn(10)

        # Create a custom loss that returns NaN
        class NaNLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.reduction = "mean"

            def forward(self, pred, target):
                return torch.tensor(float("nan"))

        nan_loss_fn = NaNLoss()

        loss = compute_loss(beta, nan_loss_fn, dataloader)
        assert torch.isnan(loss)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_batched_vs_full_equivalence(self, mock_loss_fn):
        """Test that batched and full dataset give same results."""
        X = torch.randn(100, 10)
        y = torch.randn(100)
        beta = torch.randn(10)

        # Full dataset with Dataset (fast path)
        dataset = Dataset(X, y)
        dataloader_full = DataLoader(dataset, batch_size=32, shuffle=False)
        full_loss = compute_loss(beta, mock_loss_fn, dataloader_full)

        # Batched dataset with BatchedDataset subclass (slow path)
        class MockBatchedDataset(BatchedDataset):
            def __init__(self, X, y):
                self._X = X
                self._y = y

            def __len__(self):
                return len(self._X)

            def __getitem__(self, idx):
                return self._X[idx], self._y[idx]

            @property
            def feature_dimension(self):
                return self._X.shape[1]

            @property
            def target_dimension(self):
                return 1 if self._y.dim() == 1 else self._y.shape[1]

        batched_dataset = MockBatchedDataset(X, y)
        dataloader_batched = DataLoader(batched_dataset, batch_size=32, shuffle=False)
        batched_loss = compute_loss(beta, mock_loss_fn, dataloader_batched)

        assert torch.allclose(full_loss, batched_loss, rtol=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 10, 32, 100])
    def test_various_batch_sizes(self, batch_size, mock_loss_fn):
        """Test that various batch sizes work correctly."""
        X = torch.randn(100, 10)
        y = torch.randn(100)
        beta = torch.randn(10)

        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss = compute_loss(beta, mock_loss_fn, dataloader)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
