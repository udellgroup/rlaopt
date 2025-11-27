from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import torch

from rlaopt.atoms import Atom
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.utils.device_utils import move_to_source_device

from ._inv_link_function import InverseLinkType, get_inverse_link_function
from ._loss_factory import LossType, get_loss_function


class BaseLinearModel(Atom, ABC):
    """Base class for all Linear Models."""

    def __init__(
        self,
        loss_type: LossType,
        inv_link_type: InverseLinkType,
        beta: Variable,
        dataloader: DataLoader,
        fit_intercept: bool = True,
        **loss_kwargs,
    ):
        feature_dim = dataloader.dataset.feature_dimension
        if loss_type in [LossType.LOGISTIC, LossType.MULTINOMIAL]:
            target_dim = int(dataloader.y.max().item())
            if loss_type == LossType.MULTINOMIAL:
                target_dim += 1
        else:
            target_dim = dataloader.dataset.target_dimension

        beta_value = beta.forward()
        beta_shape = beta_value.shape

        # Validate beta is consistent with dataset in dataloader
        expected_shape = (
            (
                feature_dim,
                target_dim,
            )
            if target_dim > 1
            else (feature_dim,)
        )
        if beta_shape != expected_shape:
            raise ValueError(
                f"Expected beta.shape={expected_shape} to match dataset dimensions "
                f"(feature_dim={feature_dim}), target_dim={target_dim}, "
                f"but got beta.shape={beta_shape}"
            )

        # Add weights to exprs dict
        exprs = {"beta": beta}
        # For checking beta is indeed a variable.
        variable_names = ["beta"]

        # Add intercept if specified
        if fit_intercept:
            intercept_ = Variable(
                (target_dim,),
                name="intercept",
                device=beta_value.device,
                dtype=beta_value.dtype,
            )
            exprs["intercept"] = intercept_
            variable_names.append("intercept")

        super().__init__(exprs, {}, variable_names)

        _loss_fn = get_loss_function(loss_type)
        if loss_type == LossType.POISSON_GAMMA:
            self._loss_fn = _loss_fn(loss_kwargs["power"], reduction="mean")
        else:
            self._loss_fn = _loss_fn(reduction="mean", **loss_kwargs)
        self._inv_link_fn = get_inverse_link_function(inv_link_type)
        self.dataloader = dataloader
        self.fit_intercept = fit_intercept

    @abstractmethod
    def score(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> float:
        pass

    def predict(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict target values.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Predicted target values
        """
        beta_tensor, intercept_tensor = self._get_params(beta_value)
        return self._get_prediction(beta_tensor, intercept_tensor, X=X)

    def forward(self):
        beta_tensor, intercept_tensor = self._get_params()

        return self._get_loss(beta_tensor, intercept_tensor)

    def loss(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta_tensor, intercept_tensor = self._get_params(beta_value)

        return self._get_loss(
            beta_tensor,
            intercept_tensor,
            X=X,
            y=y,
        )

    def is_smooth(self):  # pragma: no cover
        return True

    def is_proxable(self):  # pragma: no cover
        return False

    def _get_loss(
        self,
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            beta: Model parameters tensor.
            intercept: Intercept tensor (can be None if fit_intercept=False).
            X: Optional input features. If None, uses training dataset.
            y: Optional targets. Required if X is provided.

        Returns:
            Loss value
        """
        # Case 1: Test data provided
        if _has_test_data(X, y):
            predictions = self._get_prediction(beta, intercept, X=X)
            return self._loss_fn(predictions, y)

        # Case 2: Training data - in-memory Dataset
        if isinstance(self.dataloader.dataset, Dataset):
            predictions = self._get_prediction(beta, intercept)
            return self._loss_fn(predictions, self.dataloader.dataset.y)

        # Case 3: Training data - BatchedDataset

        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in self.dataloader:
            loss = self._get_batch_loss(beta, intercept, X_batch, y_batch)
            total_loss += loss
            n_samples += X_batch.shape[0]

        return total_loss / n_samples

    def _get_prediction(
        self,
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate predictions using the model parameters.

        Args:
            beta: Model parameters tensor.
            intercept: Intercept tensor (can be None if fit_intercept=False).
            X: Optional input data for inference. If None, uses training dataset.

        Returns:
            Predictions as a torch.Tensor with gradients preserved for backprop.
        """
        # Case 1: Inference on new data
        if X is not None:
            move_to_source_device(X, beta)
            linear_pred = X @ beta
            if self.fit_intercept:
                linear_pred = linear_pred + intercept
            return self._inv_link_fn(linear_pred)

        # Case 2: Training dataset - fast path for full in-memory
        if isinstance(self.dataloader.dataset, Dataset):
            linear_pred = self.dataloader.dataset.X @ beta
            if self.fit_intercept:
                linear_pred = linear_pred + intercept
            return self._inv_link_fn(linear_pred)

        # Case 3: Training dataset - batched (for both Dataset and BatchedDataset)
        predictions = []
        for X_batch, _ in self.dataloader:
            batch_raw_pred, _ = self._get_batch_prediction(beta, intercept, X_batch)
            batch_pred = self._inv_link_fn(batch_raw_pred)
            predictions.append(batch_pred)
        return torch.cat(predictions, dim=0)

    def _get_batch_loss(
        self,
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss on a single batch of data.

        Args:
            beta: Model parameters tensor
            intercept: Intercept tensor (can be None if fit_intercept=False)
            X_batch: input features for the batch
            y_batch: target values for the batch

        Returns:
            Loss for the batch
        """
        pred_batch, y_batch = self._get_batch_prediction(
            beta, intercept, X_batch, y_batch
        )

        # Temporarily set reduction to 'sum' to get total loss for the batch
        original_reduction = _change_loss_reduction(self._loss_fn, "sum")
        batch_loss = self._loss_fn(pred_batch, y_batch)
        _change_loss_reduction(self._loss_fn, original_reduction)
        return batch_loss

    def _get_batch_prediction(
        self,
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict on a single batch of data.

        Args:
            beta: Model parameters tensor
            intercept: Intercept tensor (can be None if fit_intercept=False)
            X_batch: input features for the batch
            y_batch: target values for the batch (optional)

        Returns:
            Predictions for the batch
        """
        if y_batch is not None:
            X_batch, y_batch = move_to_source_device((X_batch, y_batch), beta)
        else:
            X_batch = move_to_source_device(X_batch, beta)

        linear_pred = X_batch @ beta
        if self.fit_intercept:
            linear_pred = linear_pred + intercept

        return self._inv_link_fn(linear_pred), y_batch

    def _get_params(
        self, beta_value: TensorDict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get beta and intercept parameters from model or provided TensorDict."""
        if beta_value is not None:
            beta_tensor = beta_value["beta"]
            if self.fit_intercept:
                intercept_tensor = beta_value.get("intercept")
                if intercept_tensor is None:
                    raise ValueError(
                        "Provided beta_value has no intercept key. "
                        "If fit_intercept is True, beta_value must have key"
                        "intercept."
                    )
            else:
                intercept_tensor = None
        else:
            beta_tensor = self.get_input("beta").forward()
            intercept_tensor = (
                self.get_input("intercept").forward() if self.fit_intercept else None
            )
        return beta_tensor, intercept_tensor

    def _get_target_values(self, X, y) -> torch.Tensor:
        """Get target values from test data or dataloader."""
        return y if _has_test_data(X, y) else self.dataloader.y

    def _prox(self, location, prox_scaling):
        raise NotImplementedError(
            "Proximal operator not supported for linear model atoms."
        )


def _change_loss_reduction(loss_fn: torch.nn.modules.loss._Loss, reduction: str) -> str:
    """Change the reduction method of a loss function."""
    original_reduction = loss_fn.reduction
    loss_fn.reduction = reduction
    return original_reduction


def _has_test_data(X: torch.Tensor | None, y: torch.Tensor | None) -> bool:
    if X is not None:
        if y is None:
            raise ValueError("Must provide y when X is specified")
        else:
            return True
    else:
        return False


# ===========================================================================#
# Base Classifier
# ===========================================================================#
class BaseClassifier(BaseLinearModel):
    """Linear model for classification tasks."""

    def __init__(
        self,
        loss_type,
        inv_link_type,
        beta,
        dataloader,
        fit_intercept=True,
        **loss_kwargs,
    ):
        super().__init__(
            loss_type, inv_link_type, beta, dataloader, fit_intercept, **loss_kwargs
        )

    def decision_function(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute decision function scores.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Decision function scores
        """
        beta_tensor, intercept_tensor = self._get_params(beta_value)
        return self._get_prediction(beta_tensor, intercept_tensor, X=X)

    def predict(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict class labels.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(beta_value, X)
        return torch.argmax(probs, dim=1)

    def predict_proba(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Class probabilities
        """
        logits = self.decision_function(beta_value, X)
        if logits.dim() == 1:
            # Binary classification
            pos_probs = torch.sigmoid(logits)
            neg_probs = 1 - pos_probs
            return torch.stack([neg_probs, pos_probs], dim=1)
        else:
            # Multiclass classification
            return torch.softmax(logits, dim=1)

    def predict_log_proba(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict log class probabilities.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Log class probabilities
        """
        logits = self.decision_function(beta_value, X)
        if logits.dim() == 1:
            # Binary classification: return log probs for both classes
            log_pos_probs = torch.nn.functional.logsigmoid(logits)
            log_neg_probs = torch.nn.functional.logsigmoid(
                -logits
            )  # Using identity: log(1 - sigmoid(x)) = logsigmoid(-x)
            return torch.stack([log_neg_probs, log_pos_probs], dim=1)
        else:
            # Multiclass classification: use log_softmax
            return torch.log_softmax(input=logits, dim=1)

    def score(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> float:
        """Compute classification accuracy."""
        y = self._get_target_values(X, y)
        y_pred = self.predict(beta_value, X)
        return (y_pred == y).float().mean().item()


# ===========================================================================#
# Base Regressor
# ===========================================================================#


class BaseRegressor(BaseLinearModel):
    """Linear model for regression tasks."""

    def __init__(
        self,
        loss_type,
        inv_link_type,
        beta,
        dataloader,
        fit_intercept=True,
        **loss_kwargs,
    ):
        super().__init__(
            loss_type, inv_link_type, beta, dataloader, fit_intercept, **loss_kwargs
        )

        # Build score function based on loss type.
        # Get central tendency based on loss type.
        if loss_type == LossType.LEAST_SQUARES:
            central_tendency = "mean"
        else:
            central_tendency = "median"
        # Get deviance function and build score function.
        self._score_fn = _build_regression_score_fn(self, central_tendency)

    def score(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> float:
        """Computes generalized regression score (R^2).

        Computes the generalized R^2 score as

        R_generalized^2 = 1 - (loss_model / loss_null),

        where loss_model is the loss of the model predictions and
        loss_null is the loss of a null model predicting the central
        tendency (mean or median) of the target values.

        If loss is least-squares, this corresponds to the traditional R^2 score.
        For other robust loss functions, it provides a generalized measure of fit.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.
            y: Optional target values. Required if X is provided.

        Returns:
            Generalized Regression score (R^2)
        """
        return self._score_fn(beta_value, X, y)


def _build_regression_score_fn(
    model: BaseRegressor, central_tendency_type: str
) -> Callable:
    """Build regression score function based on loss type."""
    central_tendency = partial(_get_central_tendency, type=central_tendency_type)

    def _score(
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> float:
        y_pred = model.predict(beta_value, X)
        y = model._get_target_values(X, y)
        y_null = central_tendency(y)
        loss_model, loss_null = model._loss_fn(y_pred, y), model._loss_fn(y_null, y)
        return _get_regression_score(loss_model, loss_null)

    return _score


def _get_regression_score(loss_model: torch.Tensor, loss_null: torch.Tensor) -> float:
    """Compute regression score as 1 - (loss_model / loss_null)."""
    return 1 - (loss_model / loss_null).item()


def _get_central_tendency(y: torch.Tensor, type: str) -> torch.Tensor:
    """Compute central tendency (mean or median) of target values."""
    if type == "mean":
        return torch.mean(y) * torch.ones_like(y)
    elif type == "median":
        return torch.median(y) * torch.ones_like(y)


# ===========================================================================#
# Base GLM
# ===========================================================================#


class BaseGLM(BaseLinearModel):
    """Base class for Generalized Linear Models with link functions.

    Base class for GLMs with a non-linear link function.
    Subclasses must implement the link function and its inverse.
    Examples include Poisson, Gamma, and Inverse Gaussian regression.

    """

    def __init__(
        self,
        loss_type,
        inv_link_type,
        beta,
        dataloader,
        fit_intercept=True,
        **loss_kwargs,
    ):
        super().__init__(
            loss_type, inv_link_type, beta, dataloader, fit_intercept, **loss_kwargs
        )

    def deviance_fn(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deviance between predictions and true values."""
        return 2 * (self._loss_fn(y_pred, y_true) - self._loss_fn(y_true, y_true))

    def score(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> float:
        """Compute the pseudo R-squared score based on deviance.

        Calculates a measure of model fit analogous to R-squared for linear models,
        defined as the proportion of deviance explained by the model:

            pseudo-R² = 1 - (model_deviance / null_deviance)

        where the null deviance is computed using a constant prediction (the mean
        of y_true), and the model deviance uses the fitted predictions. Values
        range from -∞ to 1, with 1 indicating perfect fit and 0 indicating the
        model performs no better than the null model.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.
            y: Optional target values. Required if X is provided.

        Returns:
            The pseudo R-squared score. Higher values indicate better fit.
        """
        y_model = self.predict(beta_value, X)
        y_true = self._get_target_values(X, y)
        y_null = _get_central_tendency(y_true, type="mean")
        model_deviance, null_deviance = (
            self.deviance_fn(y_model, y_true),
            self.deviance_fn(y_null, y_true),
        )
        return _get_regression_score(model_deviance, null_deviance)
