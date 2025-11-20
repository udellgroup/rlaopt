from abc import ABC, abstractmethod

import pytest
import torch

from rlaopt.atoms import GammaRegression, InverseGaussianRegression, PoissonRegression
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import Variable


@pytest.fixture
def glm_data():
    """Create GLM dataset with positive targets."""
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    true_beta = torch.randn(5)
    linear_pred = X @ true_beta
    # Ensure positive targets for GLMs
    y = torch.exp(linear_pred) + torch.rand(100)
    dataset = Dataset(X, y)
    return DataLoader(dataset, batch_size=32), true_beta


@pytest.fixture
def beta_variable():
    """Create beta variable for GLM."""
    return Variable(torch.randn(5), name="beta")


class BaseGLMTest(ABC):
    """Base test class for all GLM models."""

    @abstractmethod
    @pytest.fixture
    def model(self, glm_data, beta_variable):
        """Fixture to create model instance - must be implemented by subclasses."""
        pass

    def test_predict_positive(self, model):
        """Test predictions are positive (required for GLMs)."""
        pred = model.predict()
        assert (pred > 0).all()

    def test_predict_test_data(self, model):
        """Test prediction on new data."""
        X_test = torch.randn(20, 5)
        pred = model.predict(X=X_test)
        assert pred.shape == (20,)
        assert (pred > 0).all()

    def test_link_inv_link_fn_consistency(self, model):
        """Test that link and inv_link_fn are inverses."""
        pred = model.predict()
        linked = model.link_fn(pred)
        recovered = model.inv_link_fn(linked)
        assert torch.allclose(pred, recovered, rtol=1e-4)

    def test_score_returns_float(self, model):
        """Test that score returns a float."""
        score = model.score()
        assert isinstance(score, float)

    def test_deviance_is_non_negative(self, model, glm_data):
        """Test deviance is non-negative."""
        dataloader, _ = glm_data
        y_pred = model.predict()
        y_true = dataloader.dataset.y
        deviance = model.deviance_fn(y_pred, y_true)
        assert (deviance >= 0).all()

    def test_gradient_flow(self, model):
        """Test gradients propagate through loss."""
        loss = model.forward()
        loss.backward()
        grad_beta = model.get_input("beta").forward().grad
        assert grad_beta is not None
        grad_intercept = model.get_input("intercept").forward().grad
        assert grad_intercept is not None


class TestPoissonRegression(BaseGLMTest):
    """Test suite for PoissonRegression."""

    @pytest.fixture
    def model(self, glm_data, beta_variable):
        dataloader, _ = glm_data
        return PoissonRegression(beta_variable, dataloader)

    def test_inv_link_fn_is_exp(self, model):
        """Test inverse link is exponential function."""
        linear_pred = torch.randn(10)
        result = model.inv_link_fn(linear_pred)
        expected = torch.exp(linear_pred)
        assert torch.allclose(result, expected)

    def test_link_is_log(self, model):
        """Test link function is logarithm."""
        y_pred = torch.rand(10) + 1  # Positive values
        result = model.link_fn(y_pred)
        expected = torch.log(y_pred)
        assert torch.allclose(result, expected)


class TestGammaRegression(BaseGLMTest):
    """Test suite for GammaRegression."""

    @pytest.fixture
    def model(self, glm_data, beta_variable):
        dataloader, _ = glm_data
        return GammaRegression(beta_variable, dataloader)

    def test_inv_link_fn_is_exp(self, model):
        """Test inverse link is exponential function."""
        linear_pred = torch.randn(10)
        result = model.inv_link_fn(linear_pred)
        expected = torch.exp(linear_pred)
        assert torch.allclose(result, expected)

    def test_link_is_log(self, model):
        """Test link function is logarithm."""
        y_pred = torch.rand(10) + 1  # Positive values
        result = model.link_fn(y_pred)
        expected = torch.log(y_pred)
        assert torch.allclose(result, expected)


class TestInverseGaussianRegression(BaseGLMTest):
    """Test suite for InverseGaussianRegression."""

    @pytest.fixture
    def model(self, glm_data, beta_variable):
        dataloader, _ = glm_data
        return InverseGaussianRegression(beta_variable, dataloader)

    def test_inv_link_fn_is_exp(self, model):
        """Test inverse link is exponential function."""
        linear_pred = torch.randn(10)
        result = model.inv_link_fn(linear_pred)
        expected = torch.exp(linear_pred)
        assert torch.allclose(result, expected)

    def test_link_is_log(self, model):
        """Test link function is logarithm."""
        y_pred = torch.rand(10) + 1  # Positive values
        result = model.link_fn(y_pred)
        expected = torch.log(y_pred)
        assert torch.allclose(result, expected)
