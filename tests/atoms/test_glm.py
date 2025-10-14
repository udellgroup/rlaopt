"""
Pytest suite for GLM implementations.

Tests cover:
- Basic functionality (forward pass, loss computation, predictions)
- Score methods (R², D², accuracy)
- Edge cases (dimension mismatches, device handling)
- Composition with regularizers
"""

import pytest
import torch
import numpy as np

from rlaopt.expression.expression import Variable
from rlaopt.atoms import (
    LinearRegression,
    LogisticRegression,
    MultinomialRegression,
    HuberRegression,
    L1Regression,
    PoissonRegression,
    L1Norm,
    SumSquares,
)
from rlaopt.datasets import Dataset
from rlaopt.dataloader import DataLoader
from rlaopt.solvers.configs import ProxGradConfig
from rlaopt.solvers.proximal_gradient.prox_grad import ProximalGradient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def regression_data(seed):
    """Create synthetic regression dataset."""
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features)
    true_beta = torch.randn(n_features)
    y = X @ true_beta + 0.1 * torch.randn(n_samples)

    dataset = Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader, true_beta


@pytest.fixture
def binary_classification_data(seed):
    """Create synthetic binary classification dataset."""
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features)
    true_beta = torch.randn(n_features)
    logits = X @ true_beta
    y = (torch.sigmoid(logits) > 0.5).float()

    dataset = Dataset(X, y, dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader, true_beta


@pytest.fixture
def multiclass_classification_data(seed):
    """Create synthetic multiclass classification dataset."""
    n_samples, n_features, n_classes = 100, 10, 3
    X = torch.randn(n_samples, n_features)
    true_beta = torch.randn(n_features, n_classes)
    logits = X @ true_beta
    y = logits.argmax(dim=1)  # Returns Long dtype - Dataset will preserve it

    dataset = Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader, true_beta


@pytest.fixture
def count_data(seed):
    """Create synthetic count data for Poisson regression."""
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features)
    true_beta = torch.randn(n_features) * 0.1  # Keep rates reasonable
    log_rates = X @ true_beta
    y = torch.poisson(torch.exp(log_rates))

    dataset = Dataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader, true_beta


# ============================================================================
# LinearRegression Tests
# ============================================================================


class TestLinearRegression:
    """Test suite for LinearRegression."""

    def test_initialization(self, regression_data):
        """Test basic initialization."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = LinearRegression(dataloader, beta)

        assert model is not None
        assert model.var_name == beta.name
        assert model.is_smooth()
        assert not model.is_proxable()

    def test_forward(self, regression_data):
        """Test forward pass computes loss."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = LinearRegression(dataloader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # MSE is non-negative

    def test_predict(self, regression_data):
        """Test prediction on training and test data."""
        dataloader, true_beta = regression_data
        beta = Variable(10)
        model = LinearRegression(dataloader, beta)

        # Predict on training data
        preds_train = model.predict()
        assert preds_train.shape == (100,)

        # Predict on test data
        X_test = torch.randn(20, 10)
        preds_test = model.predict(X=X_test)
        assert preds_test.shape == (20,)

        # Predict with custom beta
        custom_beta = torch.randn(10)
        preds_custom = model.predict(beta=custom_beta, X=X_test)
        assert preds_custom.shape == (20,)
        expected = X_test @ custom_beta
        assert torch.allclose(preds_custom, expected)

    def test_loss(self, regression_data):
        """Test loss computation on training and test data."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = LinearRegression(dataloader, beta)

        # Loss on training data
        loss_train = model.loss()
        assert loss_train.item() >= 0

        # Loss on test data
        X_test = torch.randn(20, 10)
        y_test = torch.randn(20)
        loss_test = model.loss(X=X_test, y=y_test)
        assert loss_test.item() >= 0

    def test_score_r2(self, regression_data):
        """Test R² score computation."""
        dataloader, true_beta = regression_data
        beta = Variable(true_beta)  # Use true parameters
        model = LinearRegression(dataloader, beta)

        # Should get high R² with true parameters
        r2 = model.score()
        assert 0.9 < r2 <= 1.0  # Should be very high

        # Test on test data
        X_test = torch.randn(20, 10)
        y_test = X_test @ true_beta + 0.1 * torch.randn(20)
        r2_test = model.score(X=X_test, y=y_test)
        assert 0.8 < r2_test <= 1.0

    def test_score_poor_fit(self, regression_data):
        """Test R² with poor model fit."""
        dataloader, _ = regression_data
        beta = Variable(10)  # Random initialization
        torch.nn.init.zeros_(beta.value)  # All zeros = predict mean
        model = LinearRegression(dataloader, beta)

        # Predicting zero should give R² ≈ 0 if data is centered
        r2 = model.score()
        assert r2 <= 0.1  # Poor fit

    def test_dimension_mismatch(self, regression_data):
        """Test that dimension mismatch raises error."""
        dataloader, _ = regression_data
        beta = Variable(5)  # Wrong number of features

        with pytest.raises(ValueError, match="Dimension mismatch"):
            LinearRegression(dataloader, beta)

    def test_composition_with_regularizer(self, regression_data):
        """Test composing with L1 regularizer."""
        dataloader, _ = regression_data
        beta = Variable(10)

        # Create Lasso objective
        data_fit = LinearRegression(dataloader, beta)
        regularizer = L1Norm(beta, scaling=0.1)
        objective = data_fit + regularizer

        # Should be able to evaluate composed objective
        loss = objective.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


# ============================================================================
# LogisticRegression Tests
# ============================================================================


class TestLogisticRegression:
    """Test suite for LogisticRegression."""

    def test_initialization(self, binary_classification_data):
        """Test basic initialization."""
        dataloader, _ = binary_classification_data
        beta = Variable(10)
        model = LogisticRegression(dataloader, beta)

        assert model is not None
        assert model.is_smooth()

    def test_forward(self, binary_classification_data):
        """Test forward pass computes loss."""
        dataloader, _ = binary_classification_data
        beta = Variable(10)
        model = LogisticRegression(dataloader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Cross-entropy is non-negative

    def test_predict_proba(self, binary_classification_data):
        """Test probability prediction."""
        dataloader, _ = binary_classification_data
        beta = Variable(10)
        model = LogisticRegression(dataloader, beta)

        # Predict on training data
        probs = model.predict_proba()
        assert probs.shape == (100,)
        assert torch.all((probs >= 0) & (probs <= 1))

        # Predict on test data
        X_test = torch.randn(20, 10)
        probs_test = model.predict_proba(X=X_test)
        assert probs_test.shape == (20,)
        assert torch.all((probs_test >= 0) & (probs_test <= 1))

    def test_score_accuracy(self, binary_classification_data):
        """Test accuracy score computation."""
        dataloader, true_beta = binary_classification_data
        beta = Variable(true_beta)
        model = LogisticRegression(dataloader, beta)

        # Should get high accuracy with true parameters
        acc = model.score()
        assert 0.7 < acc <= 1.0

        # Test on test data
        X_test = torch.randn(20, 10)
        logits = X_test @ true_beta
        y_test = (torch.sigmoid(logits) > 0.5).float()
        acc_test = model.score(X=X_test, y=y_test)
        assert 0.7 < acc_test <= 1.0

    def test_score_random_predictions(self, binary_classification_data):
        """Test accuracy with random model."""
        dataloader, _ = binary_classification_data
        beta = Variable(10)
        torch.nn.init.zeros_(beta.value)  # Zero weights = random predictions
        model = LogisticRegression(dataloader, beta)

        # Should get ~50% accuracy with zero weights (all probs ≈ 0.5)
        acc = model.score()
        assert 0.4 < acc < 0.6


# ============================================================================
# MultinomialRegression Tests
# ============================================================================


class TestMultinomialRegression:
    """Test suite for MultinomialRegression."""

    def test_initialization(self, multiclass_classification_data):
        """Test basic initialization."""
        dataloader, _ = multiclass_classification_data
        beta = Variable(10, 3)
        model = MultinomialRegression(dataloader, beta)

        assert model is not None
        assert model.is_smooth()

    def test_forward(self, multiclass_classification_data):
        """Test forward pass computes loss."""
        dataloader, _ = multiclass_classification_data
        beta = Variable(10, 3)
        model = MultinomialRegression(dataloader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_predict_proba(self, multiclass_classification_data):
        """Test probability prediction for multiple classes."""
        dataloader, _ = multiclass_classification_data
        beta = Variable(10, 3)
        model = MultinomialRegression(dataloader, beta)

        # Predict on training data
        probs = model.predict_proba()
        assert probs.shape == (100, 3)
        assert torch.all((probs >= 0) & (probs <= 1))
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(100))

        # Predict on test data
        X_test = torch.randn(20, 10)
        probs_test = model.predict_proba(X=X_test)
        assert probs_test.shape == (20, 3)
        assert torch.allclose(probs_test.sum(dim=1), torch.ones(20))

    def test_score_accuracy(self, multiclass_classification_data):
        """Test multiclass accuracy score."""
        dataloader, true_beta = multiclass_classification_data
        beta = Variable(true_beta)
        model = MultinomialRegression(dataloader, beta)

        # Should get high accuracy with true parameters
        acc = model.score()
        assert 0.7 < acc <= 1.0

    def test_score_random_predictions(self, multiclass_classification_data):
        """Test accuracy with random model."""
        dataloader, _ = multiclass_classification_data
        beta = Variable(10, 3)
        torch.nn.init.zeros_(beta.value)  # Uniform probabilities
        model = MultinomialRegression(dataloader, beta)

        # Should get ~33% accuracy with 3 classes
        acc = model.score()
        assert 0.2 < acc < 0.5


# ============================================================================
# HuberRegression Tests
# ============================================================================


class TestHuberRegression:
    """Test suite for HuberRegression."""

    def test_initialization(self, regression_data):
        """Test initialization with different delta values."""
        dataloader, _ = regression_data
        beta = Variable(10)

        # Default delta
        model1 = HuberRegression(dataloader, beta)
        assert model1.delta == 1.0

        # Custom delta
        model2 = HuberRegression(dataloader, beta, delta=0.5)
        assert model2.delta == 0.5

    def test_forward(self, regression_data):
        """Test forward pass computes Huber loss."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = HuberRegression(dataloader, beta, delta=1.0)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_score_r2(self, regression_data):
        """Test R² score for Huber regression."""
        dataloader, true_beta = regression_data
        beta = Variable(true_beta)
        model = HuberRegression(dataloader, beta)

        r2 = model.score()
        assert 0.9 < r2 <= 1.0


# ============================================================================
# L1Regression Tests
# ============================================================================


class TestL1Regression:
    """Test suite for L1Regression (LAD)."""

    def test_initialization(self, regression_data):
        """Test basic initialization."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = L1Regression(dataloader, beta)

        assert model is not None
        assert model.is_smooth()

    def test_forward(self, regression_data):
        """Test forward pass computes L1 loss."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = L1Regression(dataloader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_score_r2(self, regression_data):
        """Test R² score for L1 regression."""
        dataloader, true_beta = regression_data
        beta = Variable(true_beta)
        model = L1Regression(dataloader, beta)

        r2 = model.score()
        assert 0.9 < r2 <= 1.0


# ============================================================================
# PoissonRegression Tests
# ============================================================================


class TestPoissonRegression:
    """Test suite for PoissonRegression."""

    def test_initialization(self, count_data):
        """Test basic initialization."""
        dataloader, _ = count_data
        beta = Variable(10)
        model = PoissonRegression(dataloader, beta)

        assert model is not None
        assert model.is_smooth()

    def test_forward(self, count_data):
        """Test forward pass computes Poisson loss."""
        dataloader, _ = count_data
        beta = Variable(10)
        model = PoissonRegression(dataloader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_predict_log_rates(self, count_data):
        """Test prediction returns log rates."""
        dataloader, _ = count_data
        beta = Variable(10)
        model = PoissonRegression(dataloader, beta)

        log_rates = model.predict()
        assert log_rates.shape == (100,)

        # Rates should be positive
        rates = torch.exp(log_rates)
        assert torch.all(rates > 0)

    def test_score_d2(self, count_data):
        """Test D² (deviance explained) score."""
        dataloader, true_beta = count_data
        beta = Variable(true_beta)
        model = PoissonRegression(dataloader, beta)

        d2 = model.score()
        # D² may not be as high due to Poisson noise
        assert 0.2 < d2 <= 1.0

    def test_score_handles_zeros(self, count_data):
        """Test D² computation handles y=0 correctly."""
        dataloader, _ = count_data

        # Create data with some zeros
        X = torch.randn(50, 10)
        y = torch.cat([torch.zeros(10), torch.poisson(torch.ones(40) * 2)])
        dataset = Dataset(X, y)
        dataloader_zeros = DataLoader(dataset, batch_size=32)

        beta = Variable(10)
        model = PoissonRegression(dataloader_zeros, beta)

        # Should not error with zeros
        d2 = model.score()
        assert isinstance(d2, float)
        assert not np.isnan(d2)


# ============================================================================
# Device Handling Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDeviceHandling:
    """Test that models handle device placement correctly."""

    def test_cuda_dataset(self, seed):
        """Test GLM with dataset on CUDA."""
        X = torch.randn(50, 5, device="cuda")
        y = torch.randn(50, device="cuda")
        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16)

        beta = Variable(5)
        beta.value.data = beta.value.data.cuda()

        model = LinearRegression(dataloader, beta)
        loss = model.forward()
        assert loss.device.type == "cuda"

    def test_mixed_device_inference(self, regression_data):
        """Test prediction with data on different device."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = LinearRegression(dataloader, beta)

        # Data should be moved to match beta's device
        X_test = torch.randn(10, 10)
        preds = model.predict(X=X_test)
        assert preds.device == beta.value.device


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample(self, seed):
        """Test with single sample dataset."""
        X = torch.randn(1, 5)
        y = torch.randn(1)
        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=1)

        beta = Variable(5)
        model = LinearRegression(dataloader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)

    def test_score_requires_y_with_X(self, regression_data):
        """Test that score raises error when X provided without y."""
        dataloader, _ = regression_data
        beta = Variable(10)
        model = LinearRegression(dataloader, beta)

        X_test = torch.randn(10, 10)
        with pytest.raises(ValueError, match="Must provide y when X is specified"):
            model.score(X=X_test)

    def test_large_batch_size(self, regression_data):
        """Test with batch size larger than dataset."""
        dataloader, _ = regression_data
        # Create new dataloader with large batch size
        large_batch_loader = DataLoader(
            dataloader.dataset, batch_size=1000, shuffle=False
        )

        beta = Variable(10)
        model = LinearRegression(large_batch_loader, beta)

        loss = model.forward()
        assert isinstance(loss, torch.Tensor)

    def test_integer_labels_preserved(self, seed):
        """Test that integer classification labels are preserved as Long dtype."""
        X = torch.randn(50, 5)
        y = torch.randint(0, 3, (50,))  # Integer labels

        dataset = Dataset(X, y)

        # Check that X is float32 and y is long
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.long

    def test_float_labels_preserved(self, seed):
        """Test that float regression targets remain as specified dtype."""
        X = torch.randn(50, 5)
        y = torch.randn(50)  # Float labels

        dataset = Dataset(X, y)

        # Check that both are float32
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.float32

    def test_numpy_integer_labels(self, seed):
        """Test that numpy integer arrays are preserved as Long."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)  # Numpy integers

        dataset = Dataset(X, y)

        # Check dtypes
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.long


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_lasso_regression(self, regression_data):
        """Test Lasso = LinearRegression + L1Norm."""
        dataloader, _ = regression_data
        beta = Variable(10)

        data_fit = LinearRegression(dataloader, beta)
        regularizer = L1Norm(beta, scaling=0.01)
        lasso = data_fit + regularizer

        loss = lasso.forward()
        assert loss.item() >= 0

        # Loss should be higher than unregularized
        unregularized_loss = data_fit.forward()
        assert loss.item() >= unregularized_loss.item()

    def test_ridge_regression(self, regression_data):
        """Test Ridge = LinearRegression + SumSquares."""
        dataloader, _ = regression_data
        beta = Variable(10)

        data_fit = LinearRegression(dataloader, beta)
        regularizer = SumSquares(beta)
        ridge = data_fit + 0.01 * regularizer

        loss = ridge.forward()
        assert loss.item() >= 0

    def test_elastic_net(self, regression_data):
        """Test ElasticNet = LinearRegression + L1 + L2."""
        dataloader, _ = regression_data
        beta = Variable(10)

        data_fit = LinearRegression(dataloader, beta)
        l1_penalty = L1Norm(beta, scaling=0.01)
        l2_penalty = SumSquares(beta)

        elastic_net = data_fit + l1_penalty + 0.01 * l2_penalty

        loss = elastic_net.forward()
        assert loss.item() >= 0

    def test_prox_grad(self, regression_data):
        dataloader, _ = regression_data
        beta = Variable(10)

        data_fit = LinearRegression(dataloader, beta)
        regularizer = SumSquares(beta)
        ridge = data_fit + 0.1 * regularizer

        config = ProxGradConfig(
            eta=1, max_iters=5000, use_acceleration=True, use_linesearch=True
        )
        opt = ProximalGradient(config, ridge)

        _, err = opt.solve(ridge)

        assert err.item() <= (10**0.5) * config.tol
