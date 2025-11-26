"""Tests for linear classifers"""

from abc import ABC
import pytest
import torch
import numpy as np

from rlaopt.atoms import LogisticRegression, MultinomialRegression
from rlaopt.data import Dataset, DataLoader, BatchedDataset
from rlaopt.expression import Variable
from base_linear_model_test import BaseLinearModelTest


# =============================================================================
# BatchedDataset Implementations for Testing
# =============================================================================

class BatchedBinaryClassificationDataset(BatchedDataset):
    """BatchedDataset implementation for binary classification testing."""
    
    def __init__(self, n_samples, n_features, seed=42, device=torch.device("cpu")):
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features
        self._seed = seed
        self._device = device
        
        # Pre-generate all data with the seed
        torch.manual_seed(seed)
        self._true_beta = torch.randn(n_features, device=device)
        
        # Generate all data upfront for consistency
        self._X = torch.randn(n_samples, n_features, device=device)
        logits = self._X @ self._true_beta
        probs = torch.sigmoid(logits)
        self._y = (probs > 0.5).float()
    
    def __getitem__(self, idx):
        """Return a single sample at the given index."""
        if isinstance(idx, int):
            return self._X[idx], self._y[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
    
    def __len__(self):
        """Return total number of samples."""
        return self._n_samples
    
    @property
    def feature_dimension(self):
        return self._n_features
    
    @property
    def target_dimension(self):
        return 1
    
    @property
    def num_samples(self):
        return self._n_samples


class BatchedMulticlassClassificationDataset(BatchedDataset):
    """BatchedDataset implementation for multiclass classification testing."""
    
    def __init__(self, n_samples, n_features, n_classes=3, seed=42, device=torch.device("cpu")):
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_classes = n_classes
        self._seed = seed
        self._device = device
        
        # Pre-generate all data with the seed
        torch.manual_seed(seed)
        self._true_beta = torch.randn(n_features, n_classes, device=device)
        
        # Generate all data upfront for consistency
        self._X = torch.randn(n_samples, n_features, device=device)
        logits = self._X @ self._true_beta
        self._y = torch.argmax(logits, dim=1)
    
    def __getitem__(self, idx):
        """Return a single sample at the given index."""
        if isinstance(idx, int):
            return self._X[idx], self._y[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
    
    def __len__(self):
        """Return total number of samples."""
        return self._n_samples
    
    @property
    def feature_dimension(self):
        return self._n_features
    
    @property
    def target_dimension(self):
        return 1
    
    @property
    def num_samples(self):
        return self._n_samples


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(params=[True, False], ids=["with_intercept", "no_intercept"])
def fit_intercept(request):
    """Test both with and without intercept."""
    return request.param


@pytest.fixture(params=["cpu", "cuda"], ids=["cpu", "cuda"])
def device(request):
    """Test on both CPU and CUDA if available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

@pytest.fixture
def batch_size():
    return 25


# =============================================================================
# Base Test Class
# =============================================================================

class BaseClassifierTests(BaseLinearModelTest, ABC):
    """Base test class for classifier models."""
    
    def test_score_range(self, model):
        """Test that accuracy is in valid range."""
        score = model.score()
        assert 0.0 <= score <= 1.0
    
    def test_predict_proba_returns_valid_probabilities(self, model, device):
        """Test that predict_proba returns valid probabilities."""
        probs = model.predict_proba()
        
        # All probabilities should be in [0, 1]
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        
        # Probabilities should sum to 1 for each sample
        assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0], device=device), atol=1e-6)
    
    def test_predict_log_proba_returns_valid_log_probabilities(self, model):
        """Test that predict_log_proba returns valid log probabilities."""
        log_probs = model.predict_log_proba()
        
        # All log probabilities should be <= 0
        assert torch.all(log_probs <= 0)
        
        # exp(log_probs) should match predict_proba
        probs = model.predict_proba()
        assert torch.allclose(torch.exp(log_probs), probs, rtol=1e-5)
    
    def test_predict_consistency_with_predict_proba(self, model):
        """Test that predict is consistent with argmax of predict_proba."""
        y_pred = model.predict()
        probs = model.predict_proba()
        y_pred_from_probs = torch.argmax(probs, dim=1)
        
        assert torch.all(y_pred == y_pred_from_probs)
    

# =============================================================================
# Logistic Regression Tests
# =============================================================================

class TestLogisticRegression(BaseClassifierTests):
    """Tests for LogisticRegression model."""
    
    @pytest.fixture
    def binary_classification_data(self, device):
        """Create binary classification test data."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_samples, n_features = 100, 10
        X = torch.randn(n_samples, n_features, device=device)
        true_beta = torch.randn(n_features, device=device)
        logits = X @ true_beta
        probs = torch.sigmoid(logits)
        y = (probs > 0.5).float()
        
        return X, y
    
    @pytest.fixture
    def in_memory_dataset(self, binary_classification_data):
        """Create in-memory Dataset."""
        X, y = binary_classification_data
        return Dataset(X, y)
    
    @pytest.fixture
    def batched_dataset(self, device):
        """Create BatchedDataset."""
        return BatchedBinaryClassificationDataset(
            n_samples=100,
            n_features=10,
            seed=42,
            device=device
        )
    
    @pytest.fixture(params=["in_memory", "batched"], ids=["Dataset", "BatchedDataset"])
    def dataloader(self, request, in_memory_dataset, batched_dataset, batch_size):
        """Create DataLoader for both dataset types."""
        dataset = in_memory_dataset if request.param == "in_memory" else batched_dataset
        return DataLoader(dataset, batch_size=batch_size)
    
    @pytest.fixture
    def beta_var(self, dataloader, device):
        """Create beta Variable."""
        feature_dim = dataloader.dataset.feature_dimension
        return Variable((feature_dim,), name="beta", device=device)
    
    
    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        return LogisticRegression(beta_var, dataloader, fit_intercept)
    

    def test_invalid_init(self, dataloader):
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            LogisticRegression(beta_wrong, dataloader)
    
    
    def test_predict_returns_binary_labels(self, model):
        """Test that predict returns binary labels (0 or 1)."""
        y_pred = model.predict()
        assert torch.all((y_pred == 0) | (y_pred == 1))
    
    def test_predict_proba_shape(self, model):
        """Test that predict_proba returns correct shape."""
        probs = model.predict_proba()
        dataset = model.dataloader.dataset
        expected_samples = dataset.X.shape[0] if isinstance(dataset, Dataset) else dataset.num_samples
        
        # Should have shape (n_samples, 2) for binary classification
        assert probs.shape == (expected_samples, 2)
    
    def test_decision_function_shape(self, model):
        """Test that decision_function returns correct shape."""
        scores = model.decision_function()
        dataset = model.dataloader.dataset
        expected_samples = dataset.X.shape[0] if isinstance(dataset, Dataset) else dataset.num_samples
        
        # For binary classification, should return 1D scores
        assert scores.ndim == 1
        assert scores.shape[0] == expected_samples
    
    def test_perfect_classifier_accuracy(self, device, fit_intercept):
        """Test that perfect classifier achieves accuracy = 1.0."""
        torch.manual_seed(42)
        
        n_samples, n_features = 100, 5
        X = torch.randn(n_samples, n_features, device=device)
        true_beta = torch.randn(n_features, device=device)
        logits = X @ true_beta
        y = (logits > 0).float()
        
        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset)
        beta = Variable((n_features,), name="beta", device=device)
        
        model = LogisticRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept)
        
        # Set parameters to true values
        with torch.no_grad():
            beta.value.copy_(true_beta)
            if fit_intercept:
                model.get_input("intercept").value.zero_()
        
        accuracy = model.score()
        assert accuracy == 1.0


# =============================================================================
# Multinomial Regression Tests
# =============================================================================

class TestMultinomialRegression(BaseClassifierTests):
    """Tests for MultinomialRegression model."""

    @pytest.fixture
    def multiclass_classification_data(self, device):
        """Create multiclass classification test data."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_samples, n_features, n_classes = 100, 10, 3
        X = torch.randn(n_samples, n_features, device=device)
        true_beta = torch.randn(n_features, n_classes, device=device)
        logits = X @ true_beta
        y = torch.argmax(logits, dim=1)
        
        return X, y

    @pytest.fixture
    def in_memory_dataset(self, multiclass_classification_data):
        """Create in-memory Dataset."""
        X, y = multiclass_classification_data
        return Dataset(X, y)

    @pytest.fixture
    def batched_dataset(self, device):
        """Create BatchedDataset."""
        return BatchedMulticlassClassificationDataset(
            n_samples=100,
            n_features=10,
            n_classes=3,
            seed=42,
            device=device
        )

    @pytest.fixture(params=["in_memory", "batched"], ids=["Dataset", "BatchedDataset"])
    def dataloader(self, request, in_memory_dataset, batched_dataset, batch_size):
        """Create DataLoader for both dataset types."""
        dataset = in_memory_dataset if request.param == "in_memory" else batched_dataset
        return DataLoader(dataset, batch_size=batch_size)

    @pytest.fixture
    def beta_var(self, dataloader, device):
        """Create beta Variable with correct dimensions for multiclass."""
        feature_dim = dataloader.dataset.feature_dimension
        
        # Determine number of classes
        n_classes = int(dataloader.y.max().item()) + 1
        
        return Variable((feature_dim, n_classes), name="beta", device=device)

    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        """Create MultinomialRegression model."""
        return MultinomialRegression(
            beta_var,
            dataloader,
            fit_intercept
        )
    
    def test_invalid_init(self, dataloader):
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            MultinomialRegression(beta_wrong, dataloader)

    def get_test_data(self, n_samples, device=torch.device("cpu")):
        """Generate multiclass classification test data."""
        torch.manual_seed(42)
        n_classes = 3
        X = torch.randn(n_samples, 10, device=device)
        true_beta = torch.randn(10, n_classes, device=device)
        logits = X @ true_beta
        y = torch.argmax(logits, dim=1)
        return X, y
    
    def test_loss_on_test_data(self, model, device):
        """Test loss computation on test data."""
        torch.manual_seed(42)
        n_classes = 3
        X_test = torch.randn(model.dataloader.dataset.num_samples // 5, 10, device=device)
        true_beta = torch.randn(10, n_classes, device=device)
        logits = X_test @ true_beta
        y_test = torch.argmax(logits, dim=1)

        loss = model.loss(X=X_test, y=y_test)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_predict_returns_valid_class_labels(self, model):
        """Test that predict returns valid class labels."""
        y_pred = model.predict()
        
        # Determine number of classes
        n_classes = int(model.dataloader.y.max().item()) + 1       
 
        # All predictions should be valid class indices
        assert torch.all(y_pred >= 0) and torch.all(y_pred < n_classes)

    def test_predict_proba_shape(self, model):
        """Test that predict_proba returns correct shape."""
        probs = model.predict_proba()
        dataset = model.dataloader.dataset
        expected_samples = dataset.X.shape[0] if isinstance(dataset, Dataset) else dataset.num_samples
        
        n_classes = int(model.dataloader.y.max().item()) + 1
        
        # Should have shape (n_samples, n_classes)
        assert probs.shape == (expected_samples, n_classes)

    def test_decision_function_shape(self, model):
        """Test that decision_function returns correct shape."""
        scores = model.decision_function()
        dataset = model.dataloader.dataset
        expected_samples = dataset.X.shape[0] if isinstance(dataset, Dataset) else dataset.num_samples
        
        n_classes = int(model.dataloader.y.max().item()) + 1
        
        # For multiclass, should return 2D scores (n_samples, n_classes)
        assert scores.ndim == 2
        assert scores.shape == (expected_samples, n_classes)

    def test_perfect_classifier_accuracy(self, device, fit_intercept):
        """Test that perfect classifier achieves accuracy = 1.0."""
        torch.manual_seed(42)
        
        n_samples, n_features, n_classes = 100, 5, 3
        X = torch.randn(n_samples, n_features, device=device)
        true_beta = torch.randn(n_features, n_classes, device=device)
        logits = X @ true_beta
        y = torch.argmax(logits, dim=1)
        
        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset)
        beta = Variable((n_features, n_classes), name="beta", device=device)
        
        model = MultinomialRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept)
        
        # Set parameters to true values
        with torch.no_grad():
            beta.value.copy_(true_beta)
            if fit_intercept:
                model.get_input("intercept").value.zero_()
        
        accuracy = model.score()
        assert accuracy == 1.0

    def test_five_class_problem(self, device, fit_intercept):
        """Test multinomial regression on a 5-class problem."""
        torch.manual_seed(42)
        
        n_samples, n_features, n_classes = 150, 10, 5
        X = torch.randn(n_samples, n_features, device=device)
        true_beta = torch.randn(n_features, n_classes, device=device)
        logits = X @ true_beta
        y = torch.argmax(logits, dim=1)
        
        dataset = Dataset(X, y)
        dataloader = DataLoader(dataset)
        beta = Variable((n_features, n_classes), name="beta", device=device)
        
        model = MultinomialRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept)
        
        # Test prediction shape
        y_pred = model.predict()
        assert y_pred.shape == (n_samples,)
        
        # Test probability shape
        probs = model.predict_proba()
        assert probs.shape == (n_samples, n_classes)
        
        # Test score
        accuracy = model.score()
        assert 0.0 <= accuracy <= 1.0