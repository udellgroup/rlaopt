from abc import ABC, abstractmethod

import pytest
import torch

from rlaopt.atoms import (
    HuberRegression,
    LADRegression,
    LinearRegression,
)
from rlaopt.atoms.linear_model_base.base_regressor import _get_central_tendency
from rlaopt.data import BatchedDataset, DataLoader, Dataset
from rlaopt.expression import Variable


class MockBatchedDataset(BatchedDataset):
    """Mock BatchedDataset for testing."""

    def __init__(self):
        super().__init__()
        self._num_samples = 100
        self._feature_dim = 5
        self._target_dim = 1

        self._data = torch.randn(self.num_samples, self.feature_dimension)
        self._labels = (
            torch.randn(self.num_samples, self.target_dimension)
            if self.target_dimension > 1
            else torch.randn(self.num_samples)
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
def regression_data():
    """Create simple regression dataset with known relationship."""
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    true_beta = torch.randn(5)
    y = X @ true_beta + 0.1 * torch.randn(100)
    dataset = Dataset(X, y)
    return DataLoader(dataset, batch_size=32), true_beta


@pytest.fixture
def batch_regression_data():
    torch.manual_seed(42)
    dataset = MockBatchedDataset()
    loader = DataLoader(dataset, batch_size=10)
    return loader


@pytest.fixture
def beta_variable():
    """Create beta variable for regression."""
    return Variable(torch.randn(5), name="beta")


class TestInitLogic:
    X, y = (
        torch.ones(10, 5),
        torch.ones(
            10,
        ),
    )
    dataset = Dataset(X, y)
    beta = Variable((2,), name="beta")
    with pytest.raises(ValueError, match="Expected"):
        model = LinearRegression(beta, DataLoader(dataset))


class TestCentralTendency:
    def test_invalid_central_tendency(self):
        y = torch.ones(5)
        with pytest.raises(ValueError, match="Unsupported central tendency"):
            _get_central_tendency(y, "kurtosis")


class BaseRegressorTest(ABC):
    """Base test class for all regressor models."""

    @abstractmethod
    @pytest.fixture
    def model(self, regression_data, beta_variable):
        """Fixture to create model instance - must be implemented by subclasses."""
        pass

    @abstractmethod
    @pytest.fixture
    def batched_model(self, batch_regression_data, beta_variable):
        """Fixture to create batched model instance - must be implemented by subclasses."""
        pass

    def is_smooth(self, model):
        smooth_bool = model.is_smooth()
        assert smooth_bool is True

    def is_proxable(self, model):
        is_proxable_bool = model.is_proxable()
        assert is_proxable_bool is False

    def test_predict_training_data(self, model, batched_model):
        """Test prediction on training data."""
        pred = model.predict()
        assert pred.shape == (100,)
        assert pred.requires_grad

        pred = batched_model.predict()
        assert pred.shape == (100,)
        assert pred.requires_grad

    def test_predict_test_data(self, model, batched_model):
        """Test prediction on new data."""
        X_test = torch.randn(20, 5)
        pred = model.predict(X=X_test)
        assert pred.shape == (20,)
        assert pred.requires_grad

        pred = batched_model.predict(X=X_test)
        assert pred.shape == (20,)
        assert pred.requires_grad

    def test_score_returns_float(self, model):
        """Test that score returns a float."""
        # Compute score on training data
        score = model.score()
        assert isinstance(score, float)

        # Test score works on test data
        X_test = torch.randn(20, 5)
        y_test = torch.randn(20)
        score = model.score(X=X_test, y=y_test)
        assert isinstance(score, float)

    def test_gradient_flow(self, model, batched_model):
        """Test gradients propagate through loss."""
        loss = model.forward()
        loss.backward()
        grad_beta = model.get_input("beta").forward().grad
        assert grad_beta is not None
        grad_intercept = model.get_input("intercept").forward().grad
        assert grad_intercept is not None

        loss = batched_model.forward()
        loss.backward()
        grad_beta = batched_model.get_input("beta").forward().grad
        assert grad_beta is not None
        grad_intercept = batched_model.get_input("intercept").forward().grad
        assert grad_intercept is not None


class TestLinearRegression(BaseRegressorTest):
    """Test suite for LinearRegression (OLS)."""

    @pytest.fixture
    def model(self, regression_data, beta_variable):
        dataloader, _ = regression_data
        return LinearRegression(beta_variable, dataloader)

    @pytest.fixture
    def batched_model(self, batch_regression_data, beta_variable):
        dataloader = batch_regression_data
        return LinearRegression(beta_variable, dataloader)

    def test_score_improves_with_better_fit(self, regression_data):
        """Test that score is higher for better fitting coefficients."""
        dataloader, true_beta = regression_data

        # Model with random coefficients
        random_beta = Variable(torch.randn(5), name="beta")
        model_random = LinearRegression(random_beta, dataloader)
        score_random = model_random.score()

        # Model with true coefficients
        good_beta = Variable(true_beta.clone().requires_grad_(True), name="beta")
        model_good = LinearRegression(good_beta, dataloader)
        score_good = model_good.score()

        assert score_good > score_random
        assert score_good > 0.9  # Should be close to 1


class TestLADRegression(BaseRegressorTest):
    """Test suite for LADRegression (LAD)."""

    @pytest.fixture
    def model(self, regression_data, beta_variable):
        dataloader, _ = regression_data
        return LADRegression(beta_variable, dataloader)

    @pytest.fixture
    def batched_model(self, batch_regression_data, beta_variable):
        dataloader = batch_regression_data
        return LADRegression(beta_variable, dataloader)


class TestHuberRegression(BaseRegressorTest):
    """Test suite for HuberRegression."""

    @pytest.fixture
    def model(self, regression_data, beta_variable):
        dataloader, _ = regression_data
        return HuberRegression(beta_variable, dataloader, delta=1.0)

    @pytest.fixture
    def batched_model(self, batch_regression_data, beta_variable):
        dataloader = batch_regression_data
        return HuberRegression(beta_variable, dataloader, delta=1.0)

    def test_initialization_default_delta(self, regression_data, beta_variable):
        """Test model initializes with default delta."""
        dataloader, _ = regression_data
        model = HuberRegression(beta_variable, dataloader)
        assert model.delta == 1.0

    def test_initialization_custom_delta(self, regression_data, beta_variable):
        """Test model initializes with custom delta."""
        dataloader, _ = regression_data
        model = HuberRegression(beta_variable, dataloader, delta=2.5)
        assert model.delta == 2.5

    def test_different_deltas_affect_loss(self, regression_data, beta_variable):
        """Test that delta parameter affects the loss value."""
        dataloader, _ = regression_data

        model_small_delta = HuberRegression(beta_variable, dataloader, delta=0.5)
        model_large_delta = HuberRegression(beta_variable, dataloader, delta=5.0)

        loss_small = model_small_delta.forward()
        loss_large = model_large_delta.forward()

        # Losses should differ based on delta
        assert not torch.allclose(loss_small, loss_large)
