"""Test SapphireSplit class."""

import pytest
import torch
from tensordict import assert_allclose_td

from rlaopt.atoms import Box, L1Norm, LinearRegression, LogisticRegression, SumSquares
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import SumExpression, Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.splitting.sapphire_split import SapphireSplit


@pytest.fixture
def problem_data():
    """Shared small logistic regression problem data."""
    n, p = 32, 16
    torch.manual_seed(0)
    X = torch.randn(n, p, dtype=torch.float32)
    beta_star = torch.randn(p, dtype=torch.float32)
    probs = torch.sigmoid(X @ beta_star)
    y = (probs > 0.5).float()
    beta = Variable((p), name="beta", dtype=torch.float32)
    dataset = Dataset(X, y, dtype=X.dtype)
    return beta, dataset


@pytest.fixture
def log_reg_model(problem_data):
    """Logistic regression problem fixture reused across tests."""
    beta, datatset = problem_data
    loader = DataLoader(datatset)
    model = LogisticRegression(beta, loader, fit_intercept=False)
    return model, loader


class TestInitialization:
    """Tests for SapphireSplit initialization validation."""

    def test_invalid_input(self, log_reg_model, problem_data):
        """Test that invalid objective combinations raise ValueError."""
        beta, _ = problem_data
        model, loader = log_reg_model

        # No linear model
        with pytest.raises(ValueError, match="Objective missing"):
            SapphireSplit(SumSquares(beta))

        # Multiple linear models
        with pytest.raises(
            ValueError, match="Smooth part of the objective can only have one"
        ):
            SapphireSplit(model + LinearRegression(beta, loader, fit_intercept=False))

        # Multiple non-smooth terms
        with pytest.raises(
            ValueError,
            match="Regularizer can only consist of one non-smooth expression",
        ):
            SapphireSplit(model + L1Norm(beta) + Box(beta, lower=-1.0, upper=1.0))

        # Smooth expr depends upon variable other than beta
        y = Variable((5,), name="y", dtype=torch.float32)
        with pytest.raises(
            ValueError, match="Smooth expression depends upon variables"
        ):
            SapphireSplit(model + L1Norm(beta) + SumSquares(y))

        # Non smooth expr depends upon variable other than beta
        with pytest.raises(
            ValueError, match="Non-smooth regularizer depends upon variables"
        ):
            SapphireSplit(model + Box(y, lower=-1.0, upper=1.0))


class TestProperties:
    """Tests for SapphireSplit property accessors."""

    def test_model(self, log_reg_model):
        """Test model property returns the correct LinearModel instance."""
        model, _ = log_reg_model

        split = SapphireSplit(model)

        assert split.model is not None
        assert isinstance(split.model, LogisticRegression)

    def test_f(self, problem_data, log_reg_model):
        """Test f property: None with no regularizer, SumExpression otherwise."""
        beta, _ = problem_data
        model, _ = log_reg_model

        triv_split = SapphireSplit(model)

        assert triv_split.f is None

        split = SapphireSplit(model + SumSquares(beta))
        assert split.f is not None
        assert isinstance(split.f, SumExpression)

    def test_r(self, problem_data, log_reg_model):
        """Test r property is None when no non-smooth term, L1Norm otherwise."""
        beta, _ = problem_data
        model, _ = log_reg_model

        triv_split = SapphireSplit(model)

        assert triv_split.r is None

        split = SapphireSplit(model + L1Norm(beta))
        assert split.r is not None
        assert isinstance(split.r, L1Norm)

    def test_has_non_smooth_component(self, problem_data, log_reg_model):
        """has_non_smooth_component tracks whether the non-smooth term is present."""
        beta, _ = problem_data
        model, _ = log_reg_model

        assert SapphireSplit(model).has_non_smooth_component is False
        assert SapphireSplit(model + L1Norm(beta)).has_non_smooth_component is True

    def test_other_properties(self, log_reg_model):
        """Test loader, num_samples, and variable_values properties."""
        model, _ = log_reg_model

        split = SapphireSplit(model)

        # Loader should be model loader
        assert split.loader is model.dataloader

        # Number of samples are the same as the model
        assert split.num_samples == model.dataloader.dataset.num_samples

        # Variable values should agree
        assert isinstance(split.variable_values, TensorDict)
        assert assert_allclose_td(split.variable_values, model.variable_values)


class TestOracles:
    """Tests for SapphireSplit gradient and evaluation oracles."""

    def test_evaluate(self, problem_data, log_reg_model):
        """Test evaluate computes the full objective value correctly."""
        beta, dataset = problem_data
        model, _ = log_reg_model

        obj = model + SumSquares(beta) + L1Norm(beta)

        split = SapphireSplit(obj)

        output_ = split.evaluate(model.variable_values)

        truth = _logistic_loss(beta.forward(), dataset.X, dataset.y)
        +(torch.linalg.norm(beta.forward()) ** 2) + torch.linalg.norm(
            beta.forward(), ord=1
        )

        assert torch.allclose(output_, truth)

    def test_loss(self, problem_data, log_reg_model):
        """Test batch loss includes both model loss and smooth regularizer."""
        beta, dataset = problem_data
        model, _ = log_reg_model

        obj = model + 0.5 * SumSquares(beta)

        split = SapphireSplit(obj)

        X_batch, y_batch = dataset.X[0:10], dataset.y[0:10]

        output_ = split.loss(model.variable_values, X_batch, y_batch)

        truth = (
            _logistic_loss(beta.forward(), X_batch, y_batch)
            + 0.5 * torch.linalg.norm(beta.forward()) ** 2
        )

        assert torch.allclose(output_, truth)

    def test_grad_oracles(self, problem_data, log_reg_model):
        """Test full and batch gradient oracles match analytic gradients."""
        beta, dataset = problem_data
        model, _ = log_reg_model

        obj = model + 0.5 * SumSquares(beta)

        split = SapphireSplit(obj)

        output_grad = split.grad_loss(model.variable_values)["beta"]
        true_grad = (
            _logistic_grad(beta.forward(), dataset.X, dataset.y) + beta.forward()
        )

        assert torch.allclose(output_grad, true_grad)

        X_batch, y_batch = dataset.X[0:10], dataset.y[0:10]

        batch_grad = split.batch_grad_loss(model.variable_values, X_batch, y_batch)[
            "beta"
        ]

        true_batch_grad = (
            _logistic_grad(beta.forward(), X_batch, y_batch) + beta.forward()
        )

        assert torch.allclose(batch_grad, true_batch_grad)

    def test_prox(self, problem_data, log_reg_model):
        """Test prox operator delegates correctly to the non-smooth atom."""
        torch.manual_seed(0)

        beta, dataset = problem_data
        model, _ = log_reg_model

        r = Box(beta, lower=0.0, upper=1.0)
        obj = model + r

        split = SapphireSplit(obj)

        beta_value = TensorDict({"beta": 2.0 * torch.randn(dataset.X.shape[1])})

        output_ = split.prox(beta_value, 1.0)
        truth = r.prox(beta_value, 1.0)

        assert assert_allclose_td(output_, truth)

    def test_grad_reg(self, problem_data, log_reg_model):
        """Test grad_reg returns zero when f is None, correct gradient otherwise."""
        beta, dataset = problem_data
        model, _ = log_reg_model

        # When f is None, grad_reg should return 0.0
        split_no_f = SapphireSplit(model)
        beta_value = model.variable_values
        assert split_no_f.grad_reg(beta_value) == 0.0

        # When f = 0.5 * SumSquares(beta), grad_reg should return beta
        # (i.e. grad of 0.5||beta||^2)
        obj = model + 0.5 * SumSquares(beta)
        split_with_f = SapphireSplit(obj)
        grad = split_with_f.grad_reg(beta_value)["beta"]
        assert torch.allclose(grad, beta.forward())

    def test_subsampled_hvp(self, problem_data, log_reg_model):
        """Test subsampled Hessian-vector product matches analytic Hessian."""
        torch.manual_seed(0)

        beta, dataset = problem_data
        model, _ = log_reg_model

        obj = model + 0.5 * SumSquares(beta)

        split = SapphireSplit(obj)

        X_batch, y_batch = dataset.X[0:10], dataset.y[0:10]

        Hop = split.get_subsamp_hessian_linop(
            model.variable_values, X_batch, y_batch, X_batch.device
        )
        H_true = _logistic_hessian(beta.forward(), X_batch)

        v = torch.randn((X_batch.shape[1],), dtype=X_batch.dtype)
        assert torch.allclose(Hop @ v, H_true @ v)


def _logistic_loss(beta, X, y) -> torch.Tensor:
    logits = X @ beta
    log_prob = torch.nn.functional.logsigmoid(logits)
    n = X.shape[0]
    return -1 / n * torch.sum(y * log_prob + (1 - y) * (-logits + log_prob))


def _logistic_grad(beta, X, y) -> torch.Tensor:
    n = X.shape[0]
    p = torch.sigmoid(X @ beta)
    return 1 / n * (X.T @ (p - y))


def _logistic_hessian(beta, X) -> torch.Tensor:
    n = X.shape[0]
    p = torch.sigmoid(X @ beta)
    d = p * (1 - p)
    D = torch.diag(d)
    H = 1 / n * (X.T @ (D @ X))
    return H
