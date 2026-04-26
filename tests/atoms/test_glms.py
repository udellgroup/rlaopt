"""Tests for Tweedie loss based GLMs."""

from abc import ABC

import pytest
import torch
from base_linear_model_test import BaseLinearModelTest

from rlaopt.atoms import (
    CompoundPoissonGammaRegression,
    GammaRegression,
    InverseGaussianRegression,
    PoissonRegression,
)
from rlaopt.atoms.linear_model._tweedie import tweedie_loss
from rlaopt.data import BatchedDataset, DataLoader, Dataset
from rlaopt.expression import Variable


class BatchedGLMDataset(BatchedDataset):
    """BatchedDataset implementation for binary classification testing."""

    def __init__(self, n_samples, n_features, seed=42, device=torch.device("cpu")):
        """Initialize with synthetic GLM (Tweedie) data with positive targets."""
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features
        self._seed = seed
        self._device = device

        # Pre-generate all data with the seed
        torch.manual_seed(seed)
        self._true_beta = torch.randn(n_features, device=device).clamp_(
            min=-1.0, max=1.0
        )

        # Generate all data upfront for consistency
        self._X = torch.randn(n_samples, n_features, device=device)
        self._y = torch.exp(self._X @ self._true_beta) + torch.rand(
            n_samples, device=device
        )

    def __getitem__(self, idx):
        """Return a single sample at the given index."""
        if isinstance(idx, int):
            return self._X[idx], self._y[idx], torch.tensor(idx, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __len__(self):
        """Return total number of samples."""
        return self._n_samples

    @property
    def feature_dimension(self):
        """Number of input features."""
        return self._n_features

    @property
    def target_dimension(self):
        """Target dimension (1 for scalar GLM responses)."""
        return 1

    @property
    def num_samples(self):
        """Total number of samples in the dataset."""
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
def in_memory_dataset(device):
    """Create in-memory Dataset."""
    torch.manual_seed(42)
    X = torch.randn(100, 10, device=device)
    true_beta = torch.randn(10, device=device).clamp_(min=-1.0, max=1.0)
    linear_pred = X @ true_beta
    # Ensure positive targets for GLMs
    y = torch.exp(linear_pred) + torch.rand(100, device=device)
    dataset = Dataset(X, y)
    return dataset


@pytest.fixture
def batched_dataset(device):
    """Create BatchedDataset."""
    return BatchedGLMDataset(n_samples=100, n_features=10, seed=42, device=device)


@pytest.fixture
def batch_size():
    """Default batch size for DataLoader fixtures."""
    return 25


@pytest.fixture(params=["in_memory", "batched"], ids=["Dataset", "BatchedDataset"])
def dataloader(request, in_memory_dataset, batched_dataset, batch_size):
    """Create DataLoader for both dataset types."""
    dataset = in_memory_dataset if request.param == "in_memory" else batched_dataset
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def beta_var(device):
    """Create beta variable for GLM."""
    return Variable((10,), name="beta", device=device)


class TweedieLossTest:
    """Tests for the tweedie_loss functional."""

    def test_tweedie_loss_invalid_reduction(self):  # ← Added self
        """Test that the functional form raises ValueError for invalid reduction."""
        input_ = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.5, 1.5, 2.5])

        with pytest.raises(ValueError, match="Invalid reduction mode:"):
            tweedie_loss(input_, target, power=1.5, reduction="median")

    def test_loss_with_invalid_targets(self):
        """Test that negative targets raise ValueError."""
        with pytest.raises(ValueError, match="Target values must be non-negative"):
            input_ = torch.zeros(10)
            target = -torch.ones(10)
            tweedie_loss(input_, target)

    def test_loss_with_sum_reduction(self):
        """Test tweedie loss with sum reduction."""
        input_ = torch.zeros(10)
        target = torch.ones(10)
        loss = tweedie_loss(input_, target, reduction="sum")
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar tensor

    def test_loss_without_reduction(self):
        """Test tweedie loss with no reduction."""
        input_ = torch.zeros(10)
        target = torch.ones(10)
        loss_tensor = tweedie_loss(input_, target, reduction="none")
        assert isinstance(loss_tensor, torch.Tensor)
        assert loss_tensor.shape == (10,)

    def test_loss_with_invalid_reduction(self):
        """Test tweedie loss with invalid reduction."""
        input_ = torch.zeros(1)
        target = torch.ones(1)
        with pytest.raises(ValueError, match="Invalid reduction"):
            tweedie_loss(input_, target, reduction="median")


class BaseGLMTest(BaseLinearModelTest, ABC):
    """Base test class for all GLM models."""

    def test_predict_positive(self, model):
        """Test predictions are positive (required for GLMs)."""
        pred = model.predict()
        assert (pred > 0.0).all()

    def test_predict_test_data_positive(self, model, device):
        """Test prediction on new data."""
        X_test = torch.randn(20, 10, device=device)
        pred = model.predict(X=X_test)
        assert pred.shape == (20,)
        assert (pred > 0.0).all()

    def test_deviance_is_non_negative(self, model):
        """Test deviance is non-negative."""
        y_pred = model.predict()
        y_true = model.dataloader.y
        deviance = model.deviance_fn(y_pred, y_true)
        assert (deviance >= 0.0).all()

    def test_inv_link_fn_is_exp(self, model, device):
        """Test inverse link is exponential function."""
        linear_pred = torch.randn(10, device=device)
        result = model._inv_link_fn(linear_pred)
        expected = torch.exp(linear_pred)
        assert torch.allclose(result, expected)


class TestCompoundPoissonGammaRegression(BaseGLMTest):
    """Test suite for CompoundPoissonGammaRegression (Tweedie)."""

    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        """Create CompoundPoissonGammaRegression model instance."""
        return CompoundPoissonGammaRegression(beta_var, dataloader, fit_intercept)

    def test_invalid_init(self, dataloader):
        """Test that mismatched beta dimensions raise ValueError."""
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            CompoundPoissonGammaRegression(beta_wrong, dataloader)

    def test_invalid_power_init(self, beta_var, dataloader):
        """Test that out-of-range power raises ValueError."""
        with pytest.raises(ValueError, match="Power must be"):
            CompoundPoissonGammaRegression(beta_var, dataloader, power=4.0)

    def test_power_property(self, model):
        """Test that power property returns a float."""
        power = model.power
        assert isinstance(power, float)


class TestPoissonRegression(BaseGLMTest):
    """Test suite for PoissonRegression."""

    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        """Create PoissonRegression model instance."""
        return PoissonRegression(beta_var, dataloader, fit_intercept)

    def test_invalid_init(self, dataloader):
        """Test that mismatched beta dimensions raise ValueError."""
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            PoissonRegression(beta_wrong, dataloader)


class TestGammaRegression(BaseGLMTest):
    """Test suite for GammaRegression."""

    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        """Create GammaRegression model instance."""
        return GammaRegression(beta_var, dataloader, fit_intercept)

    def test_invalid_init(self, dataloader):
        """Test that mismatched beta dimensions raise ValueError."""
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            GammaRegression(beta_wrong, dataloader)


class TestInverseGaussianRegression(BaseGLMTest):
    """Test suite for InverseGaussianRegression."""

    @pytest.fixture
    def model(self, beta_var, dataloader, fit_intercept):
        """Create InverseGaussianRegression model instance."""
        return InverseGaussianRegression(beta_var, dataloader, fit_intercept)

    def test_invalid_init(self, dataloader):
        """Test that mismatched beta dimensions raise ValueError."""
        beta_wrong = Variable((5,))
        with pytest.raises(ValueError, match="Expected beta"):
            InverseGaussianRegression(beta_wrong, dataloader)
