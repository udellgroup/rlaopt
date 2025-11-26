# test_regressors.py
import pytest
import torch
import numpy as np

from rlaopt.atoms import LinearRegression, HuberRegression
from rlaopt.data import Dataset, DataLoader, BatchedDataset
from rlaopt.expression import Variable


# =============================================================================
# BatchedDataset Implementation for Testing
# =============================================================================

class BatchedRegressionDataset(BatchedDataset):
    """BatchedDataset implementation for regression testing.
    
    This implementation returns individual samples and relies on DataLoader
    for batching, making it compatible with PyTorch's standard DataLoader.
    """
    
    def __init__(self, n_samples, n_features, batch_size=32, seed=42, device=torch.device("cpu")):
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features
        self._batch_size = batch_size
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
        batch_size=20,
        seed=42,
        device=device
    )


@pytest.fixture(params=["in_memory", "batched"], ids=["Dataset", "BatchedDataset"])
def dataloader(request, in_memory_dataset, batched_dataset):
    """Create DataLoader for both dataset types."""
    dataset = in_memory_dataset if request.param == "in_memory" else batched_dataset
    return DataLoader(dataset)


@pytest.fixture
def beta(dataloader, device):
    """Create beta Variable."""
    feature_dim = dataloader.dataset.feature_dimension
    return Variable((feature_dim,), name="beta", device=device)


# =============================================================================
# Base Test Class
# =============================================================================

class BaseRegressorTests:
    """Base test class for regressor models."""
    
    @pytest.fixture
    def model(self, dataloader, beta, fit_intercept):
        """Create model instance - to be overridden by subclasses."""
        raise NotImplementedError
    
    # Tests for BaseLinearModel methods
    
    def test_forward_returns_scalar_tensor(self, model):
        """Test that forward() returns a scalar tensor."""
        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0 or (loss.ndim == 1 and loss.numel() == 1)
    
    def test_forward_computes_gradient(self, model):
        """Test that forward() creates computation graph."""
        loss = model.forward()
        loss.backward()
        
        beta = model.get_input("beta")
        assert beta.forward().grad is not None
        
        if model.fit_intercept:
            intercept = model.get_input("intercept")
            assert intercept.forward().grad is not None
    
    def test_loss_on_training_data(self, model):
        """Test loss computation on training data."""
        loss = model.loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_loss_on_test_data(self, model, device):
        """Test loss computation on test data."""
        torch.manual_seed(42)
        X_test = torch.randn(20, 10, device=device)
        y_test = torch.randn(20, device=device)
        
        loss = model.loss(X=X_test, y=y_test)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_loss_requires_y_when_x_provided(self, model):
        """Test that providing X without y raises error."""
        torch.manual_seed(42)
        X_test = torch.randn(20, 10)
        
        with pytest.raises(ValueError, match="Must provide y when X is specified"):
            model.loss(X=X_test, y=None)
    
    def test_predict_on_training_data(self, model):
        """Test prediction on training data."""
        y_pred = model.predict()
        assert isinstance(y_pred, torch.Tensor)
        
        # Handle both Dataset and BatchedDataset
        dataset = model.dataloader.dataset
        expected_samples = dataset.X.shape[0] if isinstance(dataset, Dataset) else dataset.num_samples
        assert y_pred.shape[0] == expected_samples
    
    def test_predict_on_test_data(self, model, device):
        """Test prediction on test data."""
        torch.manual_seed(42)
        X_test = torch.randn(20, 10, device=device)
        
        y_pred = model.predict(X=X_test)
        assert isinstance(y_pred, torch.Tensor)
        assert y_pred.shape[0] == 20
    
    def test_score_returns_float(self, model):
        """Test that score returns a float."""
        score = model.score()
        assert isinstance(score, float)
    
    def test_score_on_test_data(self, model, device):
        """Test score computation on test data."""
        torch.manual_seed(42)
        X_test = torch.randn(20, 10, device=device)
        y_test = torch.randn(20, device=device)
        
        score = model.score(X=X_test, y=y_test)
        assert isinstance(score, float)
    
    def test_intercept_created_when_specified(self, model, fit_intercept):
        """Test that intercept is created when fit_intercept=True."""
        if fit_intercept:
            assert isinstance(model.get_input("intercept"), Variable)
        else:
            with pytest.raises(KeyError):
                model.get_input("intercept")
        
    
    def test_is_smooth(self, model):
        """Test that model reports as smooth."""
        assert model.is_smooth() is True
    
    def test_is_proxable(self, model):
        """Test that model reports as not proxable."""
        assert model.is_proxable() is False
    
    # Tests for BaseRegressor methods
    
    def test_score_range(self, model):
        """Test that R^2 score is in valid range."""
        score = model.score()
        assert score <= 1.0
    
    
    def _create_model(self, beta, dataloader, fit_intercept):
        """Helper to create model - to be overridden by subclasses."""
        raise NotImplementedError


# =============================================================================
# Linear Regression Tests
# =============================================================================

class TestLinearRegression(BaseRegressorTests):
    """Tests for LinearRegression model."""
    
    @pytest.fixture
    def model(self, dataloader, beta, fit_intercept):
        """Create LinearRegression model."""
        return LinearRegression(
            beta=beta,
            dataloader=dataloader,
            fit_intercept=fit_intercept
        )
    
    def _create_model(self, beta, dataloader, fit_intercept):
        """Helper to create LinearRegression model."""
        return LinearRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept)
    
    
# =============================================================================
# Huber Regression Tests
# =============================================================================

class TestHuberRegression(BaseRegressorTests):
    """Tests for HuberRegression model."""
    
    @pytest.fixture
    def model(self, dataloader, beta, fit_intercept):
        """Create HuberRegression model with default delta."""
        return HuberRegression(
            beta=beta,
            dataloader=dataloader,
            fit_intercept=fit_intercept,
            delta=1.0
        )
    
    def _create_model(self, beta, dataloader, fit_intercept):
        """Helper to create HuberRegression model."""
        return HuberRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept, delta=1.0)
    
    def test_delta_property(self, model):
        """Test that delta property returns correct value."""
        assert model.delta == 1.0
    
    def test_default_delta(self, dataloader, beta, fit_intercept):
        """Test that default delta is 1.0."""
        model = HuberRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept)
        assert model.delta == 1.0
    
    def test_different_delta_values_affect_loss(self, device, fit_intercept):
        """Test that different delta values produce different losses."""
        torch.manual_seed(42)
        
        X = torch.randn(100, 10, device=device)
        y = torch.randn(100, device=device)
        
        losses = {}
        for delta in [0.5, 1.0, 2.0]:
            dataset = Dataset(X, y)
            dataloader = DataLoader(dataset)
            beta = Variable((10,), name="beta", device=device)
            
            model = HuberRegression(beta=beta, dataloader=dataloader, fit_intercept=fit_intercept, delta=delta)
            
            with torch.no_grad():
                beta.value.copy_(torch.ones(10, device=device))
                if fit_intercept:
                    model.get_input("intercept").value.zero_()
            
            losses[delta] = model.loss().item()
        
        assert losses[0.5] != losses[1.0]
        assert losses[1.0] != losses[2.0]
    
    def test_huber_converges_to_mse_for_large_delta(self, device, fit_intercept):
        """Test that Huber loss approaches MSE as delta increases."""
        torch.manual_seed(42)
        
        X = torch.randn(100, 10, device=device)
        y = torch.randn(100, device=device)
        
        # Huber model with very large delta
        dataset_huber = Dataset(X, y)
        dataloader_huber = DataLoader(dataset_huber)
        beta_huber = Variable((10,), name="beta", device=device)
        model_huber = HuberRegression(beta=beta_huber, dataloader=dataloader_huber, 
                                       fit_intercept=fit_intercept, delta=1000.0)
        
        # OLS model
        dataset_ols = Dataset(X, y)
        dataloader_ols = DataLoader(dataset_ols)
        beta_ols = Variable((10,), name="beta", device=device)
        model_ols = LinearRegression(beta=beta_ols, dataloader=dataloader_ols, fit_intercept=fit_intercept)
        
        # Set same parameters
        with torch.no_grad():
            params = torch.randn(10, device=device)
            beta_huber.value.copy_(params)
            beta_ols.value.copy_(params)
            
            if fit_intercept:
                intercept_val = torch.randn(1, device=device)
                model_huber.get_input("intercept").value.copy_(intercept_val)
                model_ols.get_input("intercept").value.copy_(intercept_val)
        
        loss_huber = model_huber.loss()
        loss_ols = model_ols.loss()
        
        # Huber with large delta should be 0.5 * MSE
        assert torch.allclose(loss_huber, 0.5 * loss_ols, rtol=0.01)