from abc import ABC, abstractmethod

import pytest
import torch

from rlaopt.atoms import Box
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import Variable
from rlaopt.solvers import ProxGrad, ProxGradConfig


def _generate_test_data(
    dataloader: DataLoader, device: torch.device
) -> tuple[torch.Tensor]:
    n, p = dataloader.dataset.num_samples, dataloader.dataset.feature_dimension
    n_test = n // 5
    X_test, y_test = (
        torch.randn((n_test, p), device=device),
        torch.randn(n_test, device=device),
    )
    return X_test, torch.abs(y_test)


class BaseLinearModelTest(ABC):
    @pytest.fixture
    @abstractmethod
    def model(self, beta_var, dataloader, fit_intercept): ...

    def test_forward(self, model):
        """Test that forward() returns a scalar tensor."""
        loss = model.forward()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0 and loss.numel() == 1

    def test_forward_computes_gradient(self, model):
        """Test that forward() creates computation graph."""
        loss = model.forward()
        loss.backward()

        beta = model.get_input("beta")
        assert beta.forward().grad is not None

        if model.fit_intercept:
            intercept = model.get_input("intercept")
            assert intercept.forward().grad is not None

    @abstractmethod
    def test_invalid_init(self, dataloader): ...

    def test_loss_on_training_data(self, model):
        """Test loss computation on training data."""
        loss = model.loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_on_test_data(self, model, device):
        """Test loss computation on test data."""
        torch.manual_seed(42)
        X_test, y_test = _generate_test_data(model.dataloader, device)
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
        expected_samples = (
            dataset.X.shape[0] if isinstance(dataset, Dataset) else dataset.num_samples
        )
        assert y_pred.shape[0] == expected_samples

    def test_predict_on_test_data(self, model, device):
        """Test prediction on test data."""
        X_test, _ = _generate_test_data(model.dataloader, device)

        y_pred = model.predict(X=X_test)
        assert isinstance(y_pred, torch.Tensor)
        assert y_pred.shape[0] == 20

    def test_score_returns_float(self, model):
        """Test that score returns a float."""
        score = model.score()
        assert isinstance(score, float)

    def test_score_on_test_data(self, model, device):
        """Test score computation on test data."""
        X_test, y_test = _generate_test_data(model.dataloader, device)

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

    def test_opt_integration(self, model, beta_var):
        """Tests linear models are compatible with optimizers."""
        obj_ = model + Box(beta_var, lower=-1.0, upper=1.0)

        config = ProxGradConfig(use_acceleration=True)
        opt = ProxGrad(obj_, config)

        beta_val = obj_.variable_values
        state = opt.init_state(beta_val)
        tol = 1e-2
        for _ in range(2000):
            beta_val, state = opt.step(beta_val, state)
            err = state.err.item()
            if err <= tol:
                break
        assert err <= tol

    def test_prox(self, model):
        with pytest.raises(NotImplementedError, match="Proximal operator"):
            model.prox(model.variable_values, 1.0)
