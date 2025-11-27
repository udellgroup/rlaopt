"""Concrete implementations of Linear Models.

This module provides ready-to-use linear model classes for common regression and
classification tasks, including linear regression, logistic regression,
and robust regression methods.
"""

from rlaopt.atoms.linear_model._base import BaseClassifier, BaseGLM, BaseRegressor
from rlaopt.data import DataLoader
from rlaopt.expression import Variable

from ._inv_link_function import InverseLinkType
from ._loss_factory import LossType


class HuberRegression(BaseRegressor):
    """Huber regression model (robust to outliers).

    Huber regression combines the best properties of L2 (least squares) and L1
    (absolute deviation) losses. It is quadratic for small residuals and linear
    for large residuals, making it robust to outliers while maintaining efficiency
    for normally distributed errors.

    The Huber loss is defined as:
        L(r) = 0.5 * r^2                if |r| <= delta
        L(r) = delta * (|r| - 0.5*delta) if |r| > delta

    Attributes:
        delta: The Huber loss threshold parameter.
    """

    def __init__(
        self,
        beta: Variable,
        dataloader: DataLoader,
        fit_intercept=True,
        delta: float = 1.0,
    ):
        """Initialize Huber regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and targets.
            fit_intercept: Whether to fit an intercept term. Defaults to True.
            delta: Threshold parameter that defines the point where the loss transitions
                from quadratic to linear. Smaller values increase robustness to outliers.
                Defaults to 1.0.
        """
        super().__init__(
            LossType.HUBER,
            InverseLinkType.IDENTITY,
            beta,
            dataloader,
            fit_intercept,
            delta=delta,
        )
        self._delta = delta

    @property
    def delta(self):
        """Returns Huber loss threshold parameter."""
        return self._delta


class LinearRegression(BaseRegressor):
    """Ordinary least squares (OLS) linear regression model.

    Linear regression models the relationship between features and a continuous
    target variable by minimizing the sum of squared residuals. This is the most
    common form of regression analysis.

    The squared error loss is defined as:
        L(y, ŷ) = (y - ŷ)^2
    """

    def __init__(
        self, beta: Variable, dataloader: DataLoader, fit_intercept: bool = True
    ):
        """Initialize linear regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and targets.
            fit_intercept: Whether to fit an intercept term. Defaults to True.
        """
        super().__init__(
            LossType.LEAST_SQUARES,
            InverseLinkType.IDENTITY,
            beta,
            dataloader,
            fit_intercept,
        )


# ===========================================================================#
# Classification Model Classes
# ===========================================================================#

# Logistic and Multinomial get assigned the identity link as PyTorch's
# built in losses already handle applying the inverse_link function
# as they work with the logits.


class LogisticRegression(BaseClassifier):
    """Binary logistic regression model.

    Logistic regression is used for binary classification tasks. It models the
    probability that an instance belongs to the positive class using the logistic
    (sigmoid) function.

    The binary cross-entropy loss is defined as:
        L(y, p) = -[y * log(p) + (1-y) * log(1-p)]

    where p = sigmoid(X @ β).
    """

    def __init__(
        self, beta: Variable, dataloader: DataLoader, fit_intercept: bool = True
    ):
        """Initialize logistic regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and binary
                targets (0 or 1).
            fit_intercept: Whether to fit an intercept term. Defaults to True.
        """
        super().__init__(
            LossType.LOGISTIC, InverseLinkType.IDENTITY, beta, dataloader, fit_intercept
        )


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
    """

    def __init__(
        self, beta: Variable, dataloader: DataLoader, fit_intercept: bool = True
    ):
        """Initialize multinomial regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
                Shape should be (n_features, n_classes) for multi-class classification.
            dataloader: DataLoader containing the training data with features and class
                labels (integers from 0 to n_classes-1).
            fit_intercept: Whether to fit an intercept term. Defaults to True.
        """
        super().__init__(
            LossType.MULTINOMIAL,
            InverseLinkType.IDENTITY,
            beta,
            dataloader,
            fit_intercept,
        )


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

    Note:
        Unlike pure Poisson (p=1) or Gamma (p=2) regression, the Compound
        Poisson-Gamma with p in (1, 2) handles the mixed discrete-continuous nature
        of the data with a moderate mean-variance relationship.

    Attributes:
        power: The power parameter used to define the loss.
    """

    def __init__(
        self,
        beta: Variable,
        dataloader: DataLoader,
        fit_intercept: bool = True,
        power: float = 1.5,
    ):
        """Initialize Compound Poisson-Gamma regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and
                non-negative continuous targets (may include zeros).
            fit_intercept: Whether to fit an intercept term. Defaults to True.
            power: Power parameter p ∈ (1, 2) defining the variance function.
                Defaults to 1.5.
        """
        super().__init__(
            LossType.POISSON_GAMMA,
            InverseLinkType.EXP,
            beta,
            dataloader,
            fit_intercept,
            power=power,
        )
        self._power = power

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
    """

    def __init__(
        self, beta: Variable, dataloader: DataLoader, fit_intercept: bool = True
    ):
        """Initialize Gamma regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and
                positive continuous targets.
            fit_intercept: Whether to fit an intercept term. Defaults to True.
        """
        super().__init__(
            LossType.GAMMA, InverseLinkType.EXP, beta, dataloader, fit_intercept
        )


class InverseGaussianRegression(BaseGLM):
    """Inverse Gaussian regression model with log link function.

    Inverse Gaussian regression is used for modeling continuous, positive-valued
    target variables that are right-skewed. It assumes the target variable follows
    an Inverse Gaussian distribution and uses a log link function to ensure
    predictions are positive.

    The Inverse Gaussian loss (negative log-likelihood) is defined as:
       L(y, ŷ) = (y - ŷ)^2 / (2 * y * ŷ^2)

    where ŷ = exp(X @ β) is the predicted mean.
    """

    def __init__(
        self, beta: Variable, dataloader: DataLoader, fit_intercept: bool = True
    ):
        """Initialize Inverse Gaussian regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and
                positive continuous targets.
            fit_intercept: Whether to fit an intercept term. Defaults to True.
        """
        super().__init__(
            LossType.INV_GAUSS, InverseLinkType.EXP, beta, dataloader, fit_intercept
        )


class PoissonRegression(BaseGLM):
    """Poisson regression model with log link function.

    Poisson regression is used for modeling count data and contingency tables.
    It assumes the target variable follows a Poisson distribution and uses a
    log link function to ensure predictions are positive.

    The Poisson loss (negative log-likelihood) is defined as:
        L(y, ŷ) = ŷ - y * log(ŷ)

    where ŷ = exp(X @ β) is the predicted rate parameter.
    """

    def __init__(
        self, beta: Variable, dataloader: DataLoader, fit_intercept: bool = True
    ):
        """Initialize Poisson regression model.

        Args:
            beta: Model parameters variable representing regression coefficients.
            dataloader: DataLoader containing the training data with features and count
                targets (must be non-negative).
            fit_intercept: Whether to fit an intercept term. Defaults to True.
        """
        super().__init__(
            LossType.POISSON, InverseLinkType.EXP, beta, dataloader, fit_intercept
        )
