"""
Concrete implementations of Generalized Linear Models (GLMs).

This module provides ready-to-use GLM classes for common regression and
classification tasks, including linear regression, logistic regression,
and robust regression methods.
"""

import torch
from rlaopt.expression.expression import Variable
from rlaopt.dataloader import DataLoader
from ._glm.base import _BaseGLM, _GLMClassifier
from ._glm.loss_types import LossType
from ._glm.core import _has_test_data


# ============================================================================
# Helper Functions
# ============================================================================


def _get_true_values(glm: _BaseGLM | _GLMClassifier, X, y) -> torch.Tensor:
    """Get true target values from test data or dataloader."""
    return y if _has_test_data(X, y) else glm.dataloader.y


def _get_predict(
    glm: _BaseGLM | _GLMClassifier, beta=None, X=None, y=None, get_proba: bool = False
) -> torch.Tensor:
    """Get predictions, optionally as probabilities for classifiers."""
    if isinstance(glm, _GLMClassifier) and get_proba:
        return glm.predict_proba(beta, X)
    else:
        return glm.predict(beta, X)


def _compute_r_squared(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute R² coefficient of determination."""
    ss_res = torch.sum((y_hat - y_true) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return (1 - (ss_res / ss_tot)).item()


def _compute_classification_accuracy(
    y_hat: torch.Tensor, y_true: torch.Tensor
) -> float:
    """Compute classification accuracy."""
    return (y_hat == y_true).float().mean().item()


def _compute_poisson_deviance_score(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute D² (deviance explained) for Poisson regression."""
    # Predicted rates (Poisson uses log link)
    mu = torch.exp(y_hat)

    # Model deviance: 2 * Σ[y*log(y/μ) - (y-μ)]
    nonzero = y_true > 0
    dev_model = 2 * torch.sum(
        torch.where(nonzero, y_true * torch.log(y_true / mu), torch.zeros_like(y_true))
        - (y_true - mu)
    )

    # Null deviance: use mean as prediction
    mu_null = y_true.mean()
    dev_null = 2 * torch.sum(
        torch.where(
            nonzero, y_true * torch.log(y_true / mu_null), torch.zeros_like(y_true)
        )
        - (y_true - mu_null)
    )

    return (1 - (dev_model / dev_null)).item()


# ============================================================================
# Mixin Classes for Common Scoring Methods
# ============================================================================


class _RSquaredScoringMixin:
    """Mixin providing R² scoring for regression models."""

    def score(self, beta=None, X=None, y=None) -> float:
        """Compute R² coefficient of determination.

        R² measures the proportion of variance in the target variable that is
        predictable from the features. Values range from -∞ to 1, where 1 indicates
        perfect prediction and 0 indicates the model performs no better than
        predicting the mean.

        Args:
            beta: Parameter weights for model. If None, uses the registered
                model weights. Defaults to None.
            X: Input features of shape (n_samples, n_features). If None, uses
                the training dataset. Defaults to None.
            y: Target values of shape (n_samples,). Required if X is provided.
                Defaults to None.

        Returns:
            R² score as a float.
        """
        y_hat = _get_predict(self, beta, X)
        y_true = _get_true_values(self, X, y)
        return _compute_r_squared(y_hat, y_true)


class _AccuracyScoringMixin:
    """Mixin providing accuracy scoring for classification models."""

    def score(self, beta=None, X=None, y=None) -> float:
        """Compute classification accuracy.

        Accuracy is the proportion of correct predictions, defined as the number
        of correct predictions divided by the total number of predictions. Values
        range from 0 to 1, where 1 indicates perfect classification.

        Args:
            beta: Parameter weights for model. If None, uses the registered
                model weights. Defaults to None.
            X: Input features of shape (n_samples, n_features). If None, uses
                the training dataset. Defaults to None.
            y: Target labels of shape (n_samples,). Required if X is provided.
                Defaults to None.

        Returns:
            Accuracy score as a float.
        """
        probs = _get_predict(self, beta, X, y, get_proba=True)
        y_hat = self._get_predicted_classes(probs)
        y_true = _get_true_values(self, X, y)
        return _compute_classification_accuracy(y_hat, y_true)

    def _get_predicted_classes(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert probabilities to predicted class labels."""
        raise NotImplementedError


# ============================================================================
# Regression Models
# ============================================================================


class HuberRegression(_RSquaredScoringMixin, _BaseGLM):
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


class L1Regression(_RSquaredScoringMixin, _BaseGLM):
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


class LinearRegression(_RSquaredScoringMixin, _BaseGLM):
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


class PoissonRegression(_BaseGLM):
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

    def score(self, beta=None, X=None, y=None) -> float:
        """Compute D² (deviance explained) for Poisson regression.

        D² is the Poisson equivalent of R² and measures the proportion of deviance
        explained by the model. It compares the deviance of the fitted model to the
        deviance of the null model (intercept only). Values range from -∞ to 1,
        where 1 indicates perfect prediction.

        The Poisson deviance is: 2 * Σ[y * log(y/μ) - (y - μ)]
        where μ are the predicted rates.

        Args:
            beta: Parameter weights for model. If None, uses the registered
                model weights. Defaults to None.
            X: Input features of shape (n_samples, n_features). If None, uses
                the training dataset. Defaults to None.
            y: Target count values of shape (n_samples,). Required if X is provided.
                Defaults to None.

        Returns:
            D² score as a float.
        """
        y_hat = _get_predict(self, beta, X)
        y_true = _get_true_values(self, X, y)
        return _compute_poisson_deviance_score(y_hat, y_true)


# ============================================================================
# Classification Models
# ============================================================================


class LogisticRegression(_AccuracyScoringMixin, _GLMClassifier):
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

    def predict_proba(
        self,
        beta: torch.nn.Parameter | torch.Tensor | None = None,
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
        logits = self.predict(beta, X)
        return torch.sigmoid(logits)

    def _get_predicted_classes(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert probabilities to binary class predictions."""
        return (probs > 0.5).float()


class MultinomialRegression(_AccuracyScoringMixin, _GLMClassifier):
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

    def predict_proba(
        self,
        beta: torch.nn.Parameter | torch.Tensor | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict class probabilities for all classes using the softmax function.

        Args:
            beta: Parameter weights for GLM. If None, uses the registered
                model weights. Defaults to None.
            X: Input features of shape (n_samples, n_features). If None, uses
                the training dataset. Defaults to None.

        Returns:
            Predicted probabilities for each class, shape (n_samples, n_classes).
            Each row sums to 1.0, representing a probability distribution over classes.
        """
        logits = self.predict(beta, X)
        return torch.softmax(logits, dim=1)

    def _get_predicted_classes(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert probabilities to predicted class labels."""
        return probs.argmax(dim=1)
