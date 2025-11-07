"""
Concrete implementations of Linear Models.

This module provides ready-to-use linear model classes for common regression and
classification tasks, including linear regression, logistic regression,
and robust regression methods.
"""

from functools import partial
from typing import Callable

import torch
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.data import DataLoader
from rlaopt.atoms._linear_model.base import _BaseLinearModel, _LinearClassifier, _has_test_data
from rlaopt.atoms._linear_model.loss_types import LossType

# ============================================================================
# Helper Functions
# ============================================================================

def build_regression_score_fn(
    model: _BaseLinearModel, central_tendency_type: str
) -> Callable:
    """Build regression score function based on loss type."""
    
    def _poisson_transform_y(y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y)
    
    def _identity_transform_y(y: torch.Tensor) -> torch.Tensor:
        return y

    def _score(
            deviance_fn: Callable,
            transform_y: Callable,
            beta_value: TensorDict | None = None, 
            X: torch.Tensor | None = None, 
            y: torch.Tensor | None = None
    )-> float:
        y_pred = model.predict(beta_value, X)
        y_pred = transform_y(y_pred)
        y = _get_target_values(model, X, y)
        dev_model, dev_null = deviance_fn(y_pred, y)
        return _get_regression_score(dev_model, dev_null)
        

    if model._loss_fn.__class__ == LossType.POISSON.value:
        deviance_fn = _get_poisson_deviance()
        score = partial(_score, deviance_fn = deviance_fn, transform_y=_poisson_transform_y)

    else:
        deviance_fn = _get_regression_deviance_fn(model._loss_fn, central_tendency_type)
        score = partial(_score, deviance_fn = deviance_fn, transform_y=_identity_transform_y)
    
    return score

def _get_target_values(model: _BaseLinearModel, X, y) -> torch.Tensor:
    """Get target values from test data or dataloader."""
    return y if _has_test_data(X, y) else model.dataloader.y

def _get_central_tendency(y: torch.Tensor, type: str)-> torch.Tensor:
    """Compute central tendency (mean or median) of target values."""
    if type == "mean":
        return torch.mean(y) * torch.ones_like(y)
    elif type == "median":
        return torch.median(y) * torch.ones_like(y)
    else:
        raise ValueError(f"Unsupported central tendency type: {type}")


def _get_classification_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> float:
    """Get classification accuracy."""
    return (y_pred == y_true).float().mean().item()


def _get_classification_score(
        model: _LinearClassifier,
        beta_value: TensorDict | None,
        X: torch.Tensor | None,
        y: torch.Tensor | None
    )-> float:
    """Compute classification score (accuracy) for linear classifiers."""
    y_pred = model.predict_classes(beta_value, X)
    y_true = _get_target_values(model, X, y)
    return _get_classification_accuracy(y_pred, y_true)


def _get_regression_score(model_deviance: torch.Tensor, null_deviance: torch.Tensor) -> float:
    """Compute regression score as 1 - (model_deviance / null_deviance)."""
    return 1 - (model_deviance / null_deviance).item()


def _get_regression_deviance_fn(loss_fn, central_tendency_type: str)-> Callable:
    central_tendency = partial(_get_central_tendency, type=central_tendency_type)
    def regression_deviance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_null = central_tendency(y_true)
        model_dev, null_dev = loss_fn(y_pred, y_true), loss_fn(y_null, y_true)
        return model_dev, null_dev
    return regression_deviance

def _get_poisson_deviance() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    
    def _compute_poisson_deviance(y: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        nonzero = y_true > 0
        return 2 * torch.sum(
        torch.where(nonzero, y_true * torch.log(y_true / y), torch.zeros_like(y_true))
        + (y - y_true)
    )
    
    def poisson_deviance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_null = _get_central_tendency(y_true, type="mean")
        return _compute_poisson_deviance(y_pred, y_true), _compute_poisson_deviance(y_null, y_true)
    
    return poisson_deviance



#===========================================================================#
# Regression Model Classes
#===========================================================================#

class HuberRegression(_BaseLinearModel):
    """Huber regression model (robust to outliers).

    Huber regression combines the best properties of L2 (least squares) and L1
    (absolute deviation) losses. It is quadratic for small residuals and linear
    for large residuals, making it robust to outliers while maintaining efficiency
    for normally distributed errors.

    The Huber loss is defined as:
        L(r) = 0.5 * r^2                if |r| <= delta
        L(r) = delta * (|r| - 0.5*delta) if |r| > delta

    Args:
        dataloader: DataLoader containing the training data with features and targets.
        beta: Model parameters variable representing regression coefficients.
        delta: Threshold parameter that defines the point where the loss transitions
            from quadratic to linear. Smaller values increase robustness to outliers.
            Defaults to 1.0.
    """

    def __init__(self, dataloader: DataLoader, beta: Variable, delta: float = 1.0):
        super().__init__(dataloader, beta, LossType.HUBER, delta=delta)
        self.delta = delta
        self._score_fn = build_regression_score_fn(self, central_tendency_type="median")
    
    def score(self, beta_value = None, X = None, y = None):
        return self._score_fn(beta_value, X, y)

class L1Regression(_BaseLinearModel):
    """Least absolute deviation (LAD) regression model.

    L1 regression minimizes the sum of absolute residuals, making it highly robust
    to outliers. Unlike least squares regression, which can be heavily influenced
    by extreme values, LAD regression gives equal weight to all residuals.

    The L1 loss is defined as:
        L(y, ŷ) = |y - ŷ|

    Args:
        dataloader: DataLoader containing the training data with features and targets.
        beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, dataloader: DataLoader, beta: Variable):
        super().__init__(dataloader, beta, LossType.L1_LOSS)
        self._score_fn = build_regression_score_fn(self, central_tendency_type="median")
    
    def score(self, beta_value = None, X = None, y = None):
        return self._score_fn(beta_value, X, y)

class LinearRegression(_BaseLinearModel):
    """Ordinary least squares (OLS) linear regression model.

    Linear regression models the relationship between features and a continuous
    target variable by minimizing the sum of squared residuals. This is the most
    common form of regression analysis.

    The squared error loss is defined as:
        L(y, ŷ) = (y - ŷ)^2

    Args:
        dataloader: DataLoader containing the training data with features and targets.
        beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, dataloader: DataLoader, beta: Variable):
        super().__init__(dataloader, beta, LossType.LEAST_SQUARES)
        self._score_fn = build_regression_score_fn(self, central_tendency_type="mean")

    def score(self, beta_value = None, X = None, y = None):
        return self._score_fn(beta_value, X, y)

class PoissonRegression(_BaseLinearModel):
    """Poisson regression model with log link function.

    Poisson regression is used for modeling count data and contingency tables.
    It assumes the target variable follows a Poisson distribution and uses a
    log link function to ensure predictions are positive.

    The Poisson loss (negative log-likelihood) is defined as:
        L(y, λ) = λ - y * log(λ) + log(y!)

    where λ = exp(X @ β) is the predicted rate parameter.

    Args:
        dataloader: DataLoader containing the training data with features and count
            targets (must be non-negative).
        beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, dataloader: DataLoader, beta: Variable):
        super().__init__(dataloader, beta, LossType.POISSON)
        self._score_fn = build_regression_score_fn(self, central_tendency_type="mean")
    
    def score(self, beta_value = None, X = None, y = None):
        return self._score_fn(beta_value, X, y)

#===========================================================================#
# Classification Model Classes
#===========================================================================#

class LogisticRegression(_LinearClassifier):
    """Binary logistic regression model.

    Logistic regression is used for binary classification tasks. It models the
    probability that an instance belongs to the positive class using the logistic
    (sigmoid) function.

    The binary cross-entropy loss is defined as:
        L(y, p) = -[y * log(p) + (1-y) * log(1-p)]

    where p = sigmoid(X @ β).

    Args:
        dataloader: DataLoader containing the training data with features and binary
            targets (0 or 1).
        beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, dataloader: DataLoader, beta: Variable):
        super().__init__(dataloader, beta, LossType.LOGISTIC)
        self._score = partial(_get_classification_score,
                                    model=self)

    def predict_proba(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict class probabilities using the sigmoid function.

        Args:
            beta: Parameter weights for model. If None, uses the registered
                model weights. Defaults to None.
            X: Input features of shape (n_samples, n_features). If None, uses
                the training dataset. Defaults to None.

        Returns:
            Predicted probabilities for the positive class, shape (n_samples,).
            Values are in the range [0, 1].
        """
        logits = self.predict(beta_value, X)
        return torch.sigmoid(logits)
    
    def predict_classes(self, beta, X = None):
        probs = self.predict_proba(beta, X)
        return (probs > 0.5).float()
    
    def score(self, beta_value = None, X = None, y = None):
        return self._score(
            beta_value=beta_value,
            X=X,
            y=y
        )
    
    
class MultinomialRegression(_LinearClassifier):
    """Multinomial (softmax) regression model for multi-class classification.

    Multinomial regression extends logistic regression to handle more than two
    classes. It uses the softmax function to model class probabilities and is
    trained using the cross-entropy loss.

    The cross-entropy loss is defined as:
        L(y, p) = -log(p_y)

    where p = softmax(X @ β) and p_y is the probability of the true class.

    Note:
        For multinomial regression, beta should be a matrix of shape
        (n_features, n_classes) to produce logits for each class.

    Args:
        dataloader: DataLoader containing the training data with features and class
            labels (integers from 0 to n_classes-1).
        beta: Model parameters variable representing regression coefficients.
            Shape should be (n_features, n_classes) for multi-class classification.
    """

    def __init__(self, dataloader: DataLoader, beta: Variable):
        super().__init__(dataloader, beta, LossType.MULTINOMIAL)
        self._score = partial(_get_classification_score,
                                    model=self)

    def predict_proba(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict class probabilities for all classes using the softmax function.

        Args:
            beta_value: TensorDict storing value of model weights. Defaults to None.
            X: Input features of shape (n_samples, n_features). If None, uses
                the training dataset. Defaults to None.

        Returns:
            Predicted probabilities for each class, shape (n_samples, n_classes).
            Each row sums to 1.0, representing a probability distribution over classes.
        """
        logits = self.predict(beta_value, X)
        return torch.softmax(logits, dim=1)
    
    def predict_classes(self, beta_value, X = None):
        probs = self.predict_proba(beta_value, X)
        return probs.argmax(dim=1)
    
    def score(self, beta_value = None, X = None, y = None):
        return self._score(
            beta_value=beta_value, 
            X=X, 
            y=y
        )
    
    
    
    


