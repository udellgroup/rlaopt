from functools import partial
from typing import Callable

import torch

from rlaopt.atoms.linear_model_base.base import _BaseLinearModel
from rlaopt.atoms.linear_model_base.loss_types import LossType

from rlaopt.ext_tensordict import TensorDict

class BaseRegressor(_BaseLinearModel):
    """Linear model for regression tasks."""

    def __init__(
            self, 
            loss_type,
            beta, 
            dataloader,
            fit_intercept = True, 
            **loss_kwargs 
    ):
        super().__init__(loss_type, beta, dataloader, fit_intercept, **loss_kwargs)
        
        # Build score function based on loss type.
        # Get central tendency based on loss type.
        if loss_type == LossType.LEAST_SQUARES:
            central_tendency = "mean"
        else:
            central_tendency = "median"
        # Get deviance function and build score function.
        self._score_fn = _build_regression_score_fn(self, central_tendency)
        
    def predict(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict target values.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Predicted target values
        """

        beta_tensor, intercept_tensor = self._get_params(beta_value)
        return BaseRegressor._get_raw_prediction(
            beta_tensor,
            intercept_tensor, 
            self.dataloader, 
            X=X,
            fit_intercept=self.fit_intercept
        )
    
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
    model: BaseRegressor, central_tendency_type: str) -> Callable:
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


def _get_regression_score(
    loss_model: torch.Tensor, loss_null: torch.Tensor
) -> float:
    """Compute regression score as 1 - (loss_model / loss_null)."""
    return 1 - (loss_model / loss_null).item()


def _get_central_tendency(y: torch.Tensor, type: str) -> torch.Tensor:
    """Compute central tendency (mean or median) of target values."""
    if type == "mean":
        return torch.mean(y) * torch.ones_like(y)
    elif type == "median":
        return torch.median(y) * torch.ones_like(y)
    else:
        raise ValueError(f"Unsupported central tendency type: {type}")