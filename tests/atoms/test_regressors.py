"""Tests for regression losses"""

import pytest
import torch
import numpy as np

from rlaopt.atoms import LinearRegression, HuberRegression
from rlaopt.data import Dataset, DataLoader, BatchedDataset
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict

from base_linear_model_test import BaseLinearModelTest

# =============================================================================
# BatchedDataset Implementation for Testing
# =============================================================================

class BatchedRegressionDataset(BatchedDataset):
    """BatchedDataset implementation for regression testing.
    
    This implementation returns individual samples and relies on DataLoader
    for batching, making it compatible with PyTorch's standard DataLoader.
    """
    
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
        self._y = self._X @ self._true_beta + 0.1 * torch.randn(n_samples, device=device)
    
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
# Fixtures
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
def regression_data(device):
    """Create regression test data."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features, device=device)
    true_beta = torch.randn(n_features, device=device)
    y = X @ true_beta + 0.1 * torch.randn(n_samples, device=device)
    
    return X, y


@pytest.fixture
def in_memory_dataset(regression_data):
    """Create in-memory Dataset."""
    X, y = regression_data
    return Dataset(X, y)


@pytest.fixture
def batched_dataset(device):
    """Create BatchedDataset."""
    return BatchedRegressionDataset(
        n_samples=100,
        n_features=10,
        seed=42,
        device=device
    )

@pytest.fixture
def batch_size():
    return 25


@pytest.fixture(params=["in_memory", "batched"], ids=["Dataset", "BatchedDataset"])
def dataloader(request, in_memory_dataset, batched_dataset, batch_size):
    """Create DataLoader for both dataset types."""
    dataset = in_memory_dataset if request.param == "in_memory" else batched_dataset
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def beta_var(dataloader, device):
    """Create beta Variable."""
    feature_dim = dataloader.dataset.feature_dimension
    return Variable((feature_dim,), name="beta", device=device)



# =============================================================================
# Linear Regression Tests
# =============================================================================

class TestLinearRegression(BaseLinearModelTest):

    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        return LinearRegression(beta_var, dataloader, fit_intercept)
    

    def test_get_params(self):
        torch.manual_seed(42)
        X, beta_star = torch.randn(20, 5), torch.randn(5)
        y = X @ beta_star + 0.1 * torch.randn(20)

        dataset = Dataset(X,y)
        dataloader = DataLoader(dataset)
        random_beta = Variable(torch.randn(5), name="beta")

        _model_intercept = LinearRegression(random_beta, dataloader)

        beta_value = _model_intercept.variable_values
        beta_tensor, intercept_tensor = _model_intercept._get_params(beta_value=beta_value)
        assert isinstance(beta_tensor, torch.Tensor)
        assert isinstance(intercept_tensor, torch.Tensor)

        beta_tensor = beta_value["beta"]
        beta_value = TensorDict({"beta": beta_tensor})
        with pytest.raises(ValueError, match="Provided beta_value"):
            beta_tensor, intercept_tensor = _model_intercept._get_params(beta_value=beta_value)
        
        _model = LinearRegression(random_beta, dataloader, fit_intercept=False)
        beta_value = _model.variable_values
        beta_tensor, intercept_tensor = _model._get_params(beta_value=beta_value)
        assert isinstance(beta_tensor, torch.Tensor)
        assert intercept_tensor is None


    def test_invalid_init(self, dataloader):
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            LinearRegression(beta_wrong, dataloader)

    def test_score_less_than_one(self, model):
        score = model.score()
        assert score <= 1.0

    def test_score_improves_with_better_fit(self):
        """Test that score is higher for better fitting coefficients."""
        torch.manual_seed(42)
        X, beta_star = torch.randn(20, 5), torch.randn(5)
        y = X @ beta_star + 0.1 * torch.randn(20)

        dataset = Dataset(X,y)
        dataloader = DataLoader(dataset)


        # Model with random coefficients
        random_beta = Variable(torch.randn(5), name="beta")
        model_random = LinearRegression(random_beta, dataloader)
        score_random = model_random.score()

        # Model with true coefficients
        good_beta = Variable(beta_star.clone().requires_grad_(True), name="beta")
        model_good = LinearRegression(good_beta, dataloader)
        score_good = model_good.score()

        assert score_good > score_random
        assert score_good > 0.9  # Should be close to 1
    

# =============================================================================
# Huber Regression Tests
# =============================================================================

class TestHuberRegression(BaseLinearModelTest):
    
    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        return HuberRegression(beta_var, dataloader, fit_intercept)

    def test_delta_property(self, model):
        """Test delta property."""
        assert model.delta is not None
        assert model.delta == 1.0
    
    def test_invalid_init(self, dataloader):
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            HuberRegression(beta_wrong, dataloader)
    
    def test_initialization_custom_delta(self, beta_var, dataloader):
        """Test model initializes with custom delta."""
        model = HuberRegression(beta_var, dataloader, delta=2.5)
        assert model.delta == 2.5

    def test_different_deltas_affect_loss(self):
        """Test that delta parameter affects the loss value."""
        torch.manual_seed(42)
        X, y = torch.randn(20, 5), torch.randn(20)
       
        dataset = Dataset(X,y)
        dataloader = DataLoader(dataset)
        beta_variable = Variable(torch.randn(5))

        model_small_delta = HuberRegression(beta_variable, dataloader, delta=0.5)
        model_large_delta = HuberRegression(beta_variable, dataloader, delta=5.0)

        loss_small = model_small_delta.forward()
        loss_large = model_large_delta.forward()

        # Losses should differ based on delta
        assert not torch.allclose(loss_small, loss_large)