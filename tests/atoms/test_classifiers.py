from abc import ABC, abstractmethod

import pytest
import torch

from rlaopt.atoms import (
    LogisticRegression,
    MultinomialRegression,
)
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import Variable


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    true_beta = torch.randn(5)
    logits = X @ true_beta
    y = (logits > 0).float()  # Changed from .long() to .float()
    dataset = Dataset(X, y)
    return DataLoader(dataset, batch_size=32), true_beta


@pytest.fixture
def multiclass_classification_data():
    """Create multiclass classification dataset."""
    torch.manual_seed(42)
    n_classes = 4
    X = torch.randn(100, 5)
    true_beta = torch.randn(5, n_classes)
    logits = X @ true_beta
    y = torch.argmax(logits, dim=1)
    dataset = Dataset(X, y)
    return DataLoader(dataset, batch_size=32), true_beta


@pytest.fixture
def beta_variable_binary():
    """Create beta variable for binary classification."""
    return Variable(torch.randn(5), name="beta")


@pytest.fixture
def beta_variable_multiclass():
    """Create beta variable for multiclass classification."""
    return Variable(torch.randn(5, 4), name="beta")


class BaseClassifierTest(ABC):
    """Base test class for all classifier models."""

    @abstractmethod
    @pytest.fixture
    def model(self):
        """Fixture to create model instance - must be implemented by subclasses."""
        pass

    @abstractmethod
    @pytest.fixture
    def n_samples(self):
        """Number of samples in the dataset."""
        pass

    @abstractmethod
    @pytest.fixture
    def n_classes(self):
        """Number of classes."""
        pass

    def test_decision_function(self, model, n_samples):
        """Test decision function returns correct shape."""
        scores = model.decision_function()
        assert scores.shape[0] == n_samples
        assert scores.requires_grad

    def test_predict_proba(self, model, n_samples, n_classes):
        """Test predicted probabilities sum to 1."""
        probs = model.predict_proba()
        assert probs.shape == (n_samples, n_classes)
        assert torch.allclose(probs.sum(dim=1), torch.ones(n_samples), atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_log_proba(self, model, n_samples, n_classes):
        """Test log probabilities are negative and consistent with probs."""
        log_probs = model.predict_log_proba()
        assert log_probs.shape == (n_samples, n_classes)
        assert (log_probs <= 0).all()

        # Check consistency with predict_proba
        probs = model.predict_proba()
        assert torch.allclose(torch.exp(log_probs), probs, atol=1e-5)

    def test_predict_returns_class_labels(self, model, n_samples, n_classes):
        """Test predict returns valid class labels."""
        predictions = model.predict()
        assert predictions.shape == (n_samples,)
        assert predictions.dtype == torch.long
        assert (predictions >= 0).all() and (predictions < n_classes).all()

    def test_score_in_valid_range(self, model):
        """Test accuracy score is between 0 and 1."""
        accuracy = model.score()
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_gradient_flow(self, model):
        """Test gradients propagate through loss."""
        loss = model.forward()
        loss.backward()
        grad_beta = model.get_input("beta").forward().grad
        assert grad_beta is not None
        grad_intercept = model.get_input("intercept").forward().grad
        assert grad_intercept is not None


class TestLogisticRegression(BaseClassifierTest):
    """Test suite for LogisticRegression."""

    @pytest.fixture
    def model(self, binary_classification_data, beta_variable_binary):
        dataloader, _ = binary_classification_data
        return LogisticRegression(beta_variable_binary, dataloader)

    @pytest.fixture
    def n_samples(self):
        return 100

    @pytest.fixture
    def n_classes(self):
        return 2

    def test_decision_function_is_1d(self, model):
        """Test binary classification returns 1D decision scores."""
        scores = model.decision_function()
        assert scores.dim() == 1

    def test_predict_test_data(self, model):
        """Test prediction on new data."""
        X_test = torch.randn(20, 5)
        predictions = model.predict(X=X_test)
        assert predictions.shape == (20,)
        assert set(predictions.tolist()).issubset({0, 1})


class TestMultinomialRegression(BaseClassifierTest):
    """Test suite for MultinomialRegression."""

    @pytest.fixture
    def model(self, multiclass_classification_data, beta_variable_multiclass):
        dataloader, _ = multiclass_classification_data
        return MultinomialRegression(beta_variable_multiclass, dataloader)

    @pytest.fixture
    def n_samples(self):
        return 100

    @pytest.fixture
    def n_classes(self):
        return 4

    def test_decision_function_is_2d(self, model):
        """Test multiclass returns 2D logits."""
        scores = model.decision_function()
        assert scores.dim() == 2
        assert scores.shape[1] == 4

    def test_predict_test_data(self, model):
        """Test prediction on new data."""
        X_test = torch.randn(20, 5)
        predictions = model.predict(X=X_test)
        assert predictions.shape == (20,)
        assert (predictions >= 0).all() and (predictions < 4).all()
