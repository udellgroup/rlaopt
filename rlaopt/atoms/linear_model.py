"""Concrete implementations of Linear Models.

This module provides ready-to-use linear model classes for common regression and
classification tasks, including linear regression, logistic regression,
and robust regression methods.
"""

import torch

from rlaopt.atoms.linear_model_base.base_classifier import BaseClassifier
from rlaopt.atoms.linear_model_base.base_glm import BaseGLM
from rlaopt.atoms.linear_model_base.base_regressor import BaseRegressor
from rlaopt.atoms.linear_model_base.loss_types import LossType

# ===========================================================================#
# Regression Model Classes
# ===========================================================================#


class HuberRegression(BaseRegressor):
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

    def __init__(self, beta, dataloader, fit_intercept=True, delta: float = 1.0):
        super().__init__(LossType.HUBER, beta, dataloader, fit_intercept, delta=delta)
        self._delta = delta

    @property
    def delta(self):
        """Returns Huber loss threshold parameter"""
        return self._delta


class LADRegression(BaseRegressor):
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

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.L1_LOSS, beta, dataloader, fit_intercept)


class LinearRegression(BaseRegressor):
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

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.LEAST_SQUARES, beta, dataloader, fit_intercept)


# ===========================================================================#
# Classification Model Classes
# ===========================================================================#


class LogisticRegression(BaseClassifier):
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

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.LOGISTIC, beta, dataloader, fit_intercept)


class MultinomialRegression(BaseClassifier):
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

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.MULTINOMIAL, beta, dataloader, fit_intercept)


# ===========================================================================#
# Generalized Linear Model (GLM) Classes
# ===========================================================================#


class CompoundPoissonGammaRegression(BaseGLM):
    """Compound Poisson-Gamma (Tweedie) regression model with log link function.

    The Compound Poisson-Gamma distribution, also known as the Tweedie distribution
    with power parameter p ∈ (1, 2), is particularly useful for modeling positive
    continuous data with a point mass at zero. This makes it ideal for scenarios
    where many observations are exactly zero, but non-zero values are continuous
    and positive.

    Common applications:
        - Insurance claims: Many policies have zero claims, non-zero claims are continuous
        - Rainfall modeling: Many days have zero rainfall, rainy days have continuous amounts
        - Customer spending: Many customers spend nothing, active customers spend varying amounts
        - Healthcare costs: Many patients incur zero costs, others have continuous expenses

    The model uses a log link function to ensure predictions are always positive:
        ŷ = exp(X @ β)

    The Tweedie distribution combines:
        - A Poisson process governing the number of events (including zero events)
        - A Gamma distribution for the size of each event when it occurs
        - Results in a distribution with Var(Y) = φμ^p where p is in (1,2)


    Args:
        beta: Model parameters variable representing regression coefficients.
        dataloader: DataLoader containing the training data with features and
            non-negative continuous targets (may include zeros).
        fit_intercept: boolean specifying whether to use an intercept.
        power: float in (1,2) specifying the power used in the loss,
        default value is 1.5


    Note:
        Unlike pure Poisson (p=1) or Gamma (p=2) regression, the Compound
        Poisson-Gamma with p in (1, 2) handles the mixed discrete-continuous nature
        of the data with a moderate mean-variance relationship.
    """

    def __init__(self, beta, dataloader, fit_intercept=True, power: float = 1.5):
        super().__init__(
            LossType.POISSON_GAMMA, beta, dataloader, fit_intercept, power=power
        )
        self._power = power

    def link_fn(self, y_pred: torch.Tensor) -> torch.Tensor:
        return log_link_fn(y_pred)

    def inv_link_fn(self, linear_pred: torch.Tensor) -> torch.Tensor:
        return inv_log_link_fn(linear_pred)

    @property
    def power(self) -> float:
        """Returns power used to define the loss."""
        return self._power


class GammaRegression(BaseGLM):
    """Gamma regression model with log link function.

    Gamma regression is used for modeling continuous, positive-valued target
    variables that are skewed. It assumes the target variable follows a Gamma
    distribution and uses a log link function to ensure predictions are
    positive.

    The Gamma loss (negative log-likelihood) is defined as:
        L(y, ŷ) = log(ŷ) + y/ŷ

    where ŷ = exp(X @ β) is the predicted mean.

    Args:
        dataloader: DataLoader containing the training data with features and
            positive continuous targets.
        beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.GAMMA, beta, dataloader, fit_intercept)

    def link_fn(self, y_pred: torch.Tensor) -> torch.Tensor:
        return log_link_fn(y_pred)

    def inv_link_fn(self, linear_pred: torch.Tensor) -> torch.Tensor:
        return inv_log_link_fn(linear_pred)


class InverseGaussianRegression(BaseGLM):
    """Inverse Gaussian regression model with log link function.

    Inverse Gaussian regression is used for modeling continuous, positive-valued
    target variables that are right-skewed. It assumes the target variable follows
    an Inverse Gaussian distribution and uses a log link function to ensure
    predictions are positive.

    The Inverse Gaussian loss (negative log-likelihood) is defined as:
       L(y, ŷ) = (y - ŷ)^2 / (2 * y * ŷ^2)

    where ŷ = exp(X @ β) is the predicted mean.

    Args:
       dataloader: DataLoader containing the training data with features and
           positive continuous targets.
       beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.INV_GAUSS, beta, dataloader, fit_intercept)

    def link_fn(self, y_pred: torch.Tensor) -> torch.Tensor:
        return log_link_fn(y_pred)

    def inv_link_fn(self, linear_pred: torch.Tensor) -> torch.Tensor:
        return inv_log_link_fn(linear_pred)


class PoissonRegression(BaseGLM):
    """Poisson regression model with log link function.

    Poisson regression is used for modeling count data and contingency tables.
    It assumes the target variable follows a Poisson distribution and uses a
    log link function to ensure predictions are positive.

    The Poisson loss (negative log-likelihood) is defined as:
        L(y, ŷ) = ŷ - y * log(ŷ)

    where ŷ = exp(X @ β) is the predicted rate parameter.

    Args:
        dataloader: DataLoader containing the training data with features and count
            targets (must be non-negative).
        beta: Model parameters variable representing regression coefficients.
    """

    def __init__(self, beta, dataloader, fit_intercept=True):
        super().__init__(LossType.POISSON, beta, dataloader, fit_intercept)

    def link_fn(self, y_pred: torch.Tensor) -> torch.Tensor:
        return log_link_fn(y_pred)

    def inv_link_fn(self, linear_pred: torch.Tensor) -> torch.Tensor:
        return inv_log_link_fn(linear_pred)


def log_link_fn(y_pred: torch.Tensor) -> torch.Tensor:
    """Compute log of the link function (i.e., linear predictor)."""
    return torch.log(y_pred)


def inv_log_link_fn(linear_pred: torch.Tensor) -> torch.Tensor:
    """Compute inverse of the log link function (i.e., predicted mean)."""
    return torch.exp(linear_pred)
