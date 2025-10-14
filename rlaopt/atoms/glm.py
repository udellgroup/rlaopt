"""
Concrete implementations of Generalized Linear Models (GLMs).

This module provides ready-to-use GLM classes for common regression and
classification tasks, including linear regression, logistic regression,
and robust regression methods.

Classes:
    HuberRegression: Robust regression using Huber loss.
    L1Regression: Least absolute deviation regression.
    LinearRegression: Ordinary least squares regression.
    PoissonRegression: Poisson regression with log link.
    LogisticRegression: Binary logistic regression.
    MultinomialRegression: Multinomial (softmax) regression.
"""

import torch
from rlaopt.expression.expression import Variable
from rlaopt.dataloader import DataLoader
from rlaopt._typing import TensorDict
from ._glm.base import _BaseGLM, _GLMClassifier
from ._glm.loss_types import LossType


class HuberRegression(_BaseGLM):
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

    Examples:
        >>> from rlaopt.datasets import Dataset
        >>> from rlaopt.dataloader import DataLoader
        >>> from rlaopt.expression import Variable
        >>> import torch
        >>>
        >>> # Create dataset and dataloader
        >>> X = torch.randn(100, 10)
        >>> y = torch.randn(100)
        >>> dataset = Dataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Initialize model
        >>> beta = Variable(10)
        >>> model = HuberRegression(dataloader, beta)
        >>>
        >>> # More robust to outliers with smaller delta
        >>> robust_model = HuberRegression(dataloader, beta, delta=0.5)
        >>>
        >>> # Compute loss
        >>> loss = model.forward()
    """

    def __init__(
        self,
        dataloader: DataLoader,
        beta: Variable,
        delta: float = 1.0,
    ):
        super().__init__(dataloader, beta, LossType.HUBER, delta=delta)
        self.delta = delta

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

        Examples:
            >>> # Score on training data
            >>> r2_train = model.score()
            >>>
            >>> # Score on test data
            >>> X_test = torch.randn(20, 10)
            >>> y_test = torch.randn(20)
            >>> r2_test = model.score(X=X_test, y=y_test)
        """
        if beta is None:
            beta = self.get_variable(self.var_name)

        if X is not None:
            if y is None:
                raise ValueError("Must provide y when X is specified")
            predictions = self.predict(beta, X)
        else:
            predictions = self.predict(beta)
            y = self.dataloader.dataset.y

        ss_res = torch.sum((y - predictions) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        return (1 - (ss_res / ss_tot)).item()


class L1Regression(_BaseGLM):
    """Least absolute deviation (LAD) regression model.

    L1 regression minimizes the sum of absolute residuals, making it highly robust
    to outliers. Unlike least squares regression, which can be heavily influenced
    by extreme values, LAD regression gives equal weight to all residuals.

    The L1 loss is defined as:
        L(y, ŷ) = |y - ŷ|

    Args:
        dataloader: DataLoader containing the training data with features and targets.
        beta: Model parameters variable representing regression coefficients.

    Examples:
        >>> from rlaopt.datasets import Dataset
        >>> from rlaopt.dataloader import DataLoader
        >>> from rlaopt.expression import Variable
        >>> import torch
        >>>
        >>> # Create dataset and dataloader
        >>> X = torch.randn(100, 10)
        >>> y = torch.randn(100)
        >>> dataset = Dataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>>
        >>> # Initialize model
        >>> beta = Variable(10)
        >>> model = L1Regression(dataloader, beta)
        >>>
        >>> # Compute loss
        >>> loss = model.forward()
        >>>
        >>> # Make predictions on test data
        >>> X_test = torch.randn(20, 10)
        >>> predictions = model.predict(X=X_test)
    """

    def __init__(
        self,
        dataloader: DataLoader,
        beta: Variable,
    ):
        super().__init__(dataloader, beta, LossType.L1_LOSS)

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

        Examples:
            >>> # Score on training data
            >>> r2_train = model.score()
            >>>
            >>> # Score on test data
            >>> X_test = torch.randn(20, 10)
            >>> y_test = torch.randn(20)
            >>> r2_test = model.score(X=X_test, y=y_test)
        """
        if beta is None:
            beta = self.get_variable(self.var_name)

        if X is not None:
            if y is None:
                raise ValueError("Must provide y when X is specified")
            predictions = self.predict(beta, X)
        else:
            predictions = self.predict(beta)
            y = self.dataloader.dataset.y

        ss_res = torch.sum((y - predictions) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        return (1 - (ss_res / ss_tot)).item()


class LinearRegression(_BaseGLM):
    """Ordinary least squares (OLS) linear regression model.

    Linear regression models the relationship between features and a continuous
    target variable by minimizing the sum of squared residuals. This is the most
    common form of regression analysis.

    The squared error loss is defined as:
        L(y, ŷ) = (y - ŷ)^2

    Args:
        dataloader: DataLoader containing the training data with features and targets.
        beta: Model parameters variable representing regression coefficients.

    Examples:
        >>> from rlaopt.datasets import Dataset
        >>> from rlaopt.dataloader import DataLoader
        >>> from rlaopt.expression import Variable
        >>> import torch
        >>>
        >>> # Create dataset and dataloader
        >>> X = torch.randn(100, 10)
        >>> y = X @ torch.randn(10) + 0.1 * torch.randn(100)  # Linear relation
        >>> dataset = Dataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Initialize model
        >>> beta = Variable(10)
        >>> model = LinearRegression(dataloader, beta)
        >>>
        >>> # Compute loss on training data
        >>> loss = model.forward()
        >>>
        >>> # Make predictions
        >>> predictions = model.predict()
        >>>
        >>> # Evaluate on test data
        >>> X_test = torch.randn(20, 10)
        >>> y_test = torch.randn(20)
        >>> test_loss = model.loss(X=X_test, y=y_test)
    """

    def __init__(
        self,
        dataloader: DataLoader,
        beta: Variable,
    ):
        super().__init__(dataloader, beta, LossType.LEAST_SQUARES)

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

        Examples:
            >>> # Score on training data
            >>> r2_train = model.score()
            >>>
            >>> # Score on test data
            >>> X_test = torch.randn(20, 10)
            >>> y_test = torch.randn(20)
            >>> r2_test = model.score(X=X_test, y=y_test)
        """
        if beta is None:
            beta = self.get_variable(self.var_name)

        if X is not None:
            if y is None:
                raise ValueError("Must provide y when X is specified")
            predictions = self.predict(beta, X)
        else:
            predictions = self.predict(beta)
            y = self.dataloader.dataset.y

        ss_res = torch.sum((y - predictions) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        return (1 - (ss_res / ss_tot)).item()


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

    Examples:
        >>> from rlaopt.datasets import Dataset
        >>> from rlaopt.dataloader import DataLoader
        >>> from rlaopt.expression import Variable
        >>> import torch
        >>>
        >>> # Create dataset with count data
        >>> X = torch.randn(100, 10)
        >>> # Generate Poisson-distributed counts
        >>> y = torch.poisson(torch.exp(X @ torch.randn(10)))
        >>> dataset = Dataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>>
        >>> # Initialize model
        >>> beta = Variable(10)
        >>> model = PoissonRegression(dataloader, beta)
        >>>
        >>> # Compute loss
        >>> loss = model.forward()
        >>>
        >>> # Make predictions (returns log rates by default)
        >>> log_rates = model.predict()
        >>> rates = torch.exp(log_rates)
    """

    def __init__(
        self,
        dataloader: DataLoader,
        beta: Variable,
    ):
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

        Examples:
            >>> # Score on training data
            >>> d2_train = model.score()
            >>>
            >>> # Score on test data
            >>> X_test = torch.randn(20, 10)
            >>> y_test = torch.poisson(torch.exp(torch.randn(20)))
            >>> d2_test = model.score(X=X_test, y=y_test)
        """
        if beta is None:
            beta = self.get_variable(self.var_name)

        if X is not None:
            if y is None:
                raise ValueError("Must provide y when X is specified")
            predictions = self.predict(beta, X)
        else:
            predictions = self.predict(beta)
            y = self.dataloader.dataset.y

        # Predicted rates (Poisson uses log link)
        mu = torch.exp(predictions)

        # Model deviance: 2 * Σ[y*log(y/μ) - (y-μ)]
        # Handle y=0 case: 0*log(0/μ) = 0
        nonzero = y > 0
        dev_model = 2 * torch.sum(
            torch.where(nonzero, y * torch.log(y / mu), torch.zeros_like(y)) - (y - mu)
        )

        # Null deviance: use mean as prediction
        mu_null = y.mean()
        dev_null = 2 * torch.sum(
            torch.where(nonzero, y * torch.log(y / mu_null), torch.zeros_like(y))
            - (y - mu_null)
        )

        return (1 - (dev_model / dev_null)).item()


class LogisticRegression(_GLMClassifier):
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

    Examples:
        >>> from rlaopt.datasets import Dataset
        >>> from rlaopt.dataloader import DataLoader
        >>> from rlaopt.expression import Variable
        >>> import torch
        >>>
        >>> # Create dataset with binary labels
        >>> X = torch.randn(100, 10)
        >>> # Generate binary labels
        >>> y = (torch.sigmoid(X @ torch.randn(10)) > 0.5).float()
        >>> dataset = Dataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Initialize model
        >>> beta = Variable(10)
        >>> model = LogisticRegression(dataloader, beta)
        >>>
        >>> # Compute loss
        >>> loss = model.forward()
        >>>
        >>> # Get class probabilities
        >>> probs = model.predict_proba()
        >>>
        >>> # Make predictions on test data
        >>> X_test = torch.randn(20, 10)
        >>> test_probs = model.predict_proba(X=X_test)
        >>> predictions = (test_probs > 0.5).long()
    """

    def __init__(
        self,
        dataloader: DataLoader,
        beta: Variable,
    ):
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

        Examples:
            >>> # Predictions on training data
            >>> probs = model.predict_proba()
            >>>
            >>> # Predictions with custom parameters
            >>> custom_beta = torch.randn(10)
            >>> probs = model.predict_proba(beta=custom_beta)
            >>>
            >>> # Predictions on test data
            >>> X_test = torch.randn(20, 10)
            >>> test_probs = model.predict_proba(X=X_test)
            >>> predictions = (test_probs > 0.5).long()
        """
        if X is not None:
            logits = self.predict(beta, X)
        else:
            logits = self.predict(beta)
        return torch.sigmoid(logits)

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
            y: Target binary labels of shape (n_samples,). Required if X is provided.
                Defaults to None.

        Returns:
            Accuracy score as a float.

        Examples:
            >>> # Score on training data
            >>> acc_train = model.score()
            >>>
            >>> # Score on test data
            >>> X_test = torch.randn(20, 10)
            >>> y_test = torch.randint(0, 2, (20,)).float()
            >>> acc_test = model.score(X=X_test, y=y_test)
        """
        if beta is None:
            beta = self.get_variable(self.var_name)

        if X is not None:
            if y is None:
                raise ValueError("Must provide y when X is specified")
            probs = self.predict_proba(beta, X)
        else:
            probs = self.predict_proba(beta)
            y = self.dataloader.dataset.y

        predictions = (probs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        return accuracy.item()


class MultinomialRegression(_GLMClassifier):
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

    Examples:
        >>> from rlaopt.datasets import Dataset
        >>> from rlaopt.dataloader import DataLoader
        >>> from rlaopt.expression import Variable
        >>> import torch
        >>>
        >>> # Create dataset with multi-class labels
        >>> X = torch.randn(100, 10)
        >>> # Generate multi-class labels (3 classes)
        >>> y = torch.randint(0, 3, (100,))
        >>> dataset = Dataset(X, y)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>>
        >>> # Initialize model with matrix beta: n_features × n_classes
        >>> beta = Variable(10, 3)
        >>> model = MultinomialRegression(dataloader, beta)
        >>>
        >>> # Compute loss
        >>> loss = model.forward()
        >>>
        >>> # Get class probabilities
        >>> probs = model.predict_proba()  # Shape: (100, 3)
        >>> predictions = probs.argmax(dim=1)  # Predicted class labels
        >>>
        >>> # Predictions on test data
        >>> X_test = torch.randn(20, 10)
        >>> test_probs = model.predict_proba(X=X_test)
        >>> top_2_classes = test_probs.topk(2, dim=1).indices
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

        Examples:
            >>> # Predictions on training data
            >>> probs = model.predict_proba()
            >>>
            >>> # Predictions with custom parameters
            >>> custom_beta = torch.randn(10, 3)
            >>> probs = model.predict_proba(beta=custom_beta)
            >>>
            >>> # Predictions on test data
            >>> X_test = torch.randn(20, 10)
            >>> test_probs = model.predict_proba(X=X_test)  # Shape: (20, 3)
            >>> predicted_classes = test_probs.argmax(dim=1)
            >>> confidence = test_probs.max(dim=1).values
        """
        if X is not None:
            logits = self.predict(beta, X)
        else:
            logits = self.predict(beta)
        return torch.softmax(logits, dim=1)

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
            y: Target class labels of shape (n_samples,) with values in
                [0, n_classes-1]. Required if X is provided. Defaults to None.

        Returns:
            Accuracy score as a float.

        Examples:
            >>> # Score on training data
            >>> acc_train = model.score()
            >>>
            >>> # Score on test data
            >>> X_test = torch.randn(20, 10)
            >>> y_test = torch.randint(0, 3, (20,))
            >>> acc_test = model.score(X=X_test, y=y_test)
        """
        if beta is None:
            beta = self.get_variable(self.var_name)

        if X is not None:
            if y is None:
                raise ValueError("Must provide y when X is specified")
            probs = self.predict_proba(beta, X)
        else:
            probs = self.predict_proba(beta)
            y = self.dataloader.dataset.y

        predictions = probs.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        return accuracy.item()
