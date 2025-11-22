from abc import ABC, abstractmethod

import torch

from rlaopt.atoms.linear_model_base.base import _BaseLinearModel
from rlaopt.atoms.linear_model_base.base_regressor import (
    _get_central_tendency,
    _get_regression_score,
)
from rlaopt.ext_tensordict import TensorDict


class BaseGLM(_BaseLinearModel, ABC):
    """Base class for Generalized Linear Models with link functions.

    Base class for GLMs with a non-linear link function.
    Subclasses must implement the link function and its inverse.
    Examples include Poisson, Gamma, and Inverse Gaussian regression.

    """

    def __init__(self, loss_type, beta, dataloader, fit_intercept=True, **loss_kwargs):
        super().__init__(loss_type, beta, dataloader, fit_intercept, **loss_kwargs)

    @abstractmethod
    def link_fn(self, linear_pred: torch.Tensor) -> torch.Tensor:
        """Apply link function to linear predictor."""
        pass

    @abstractmethod
    def inv_link_fn(self, prediction: torch.Tensor) -> torch.Tensor:
        """Apply inverse link function."""
        pass

    def deviance_fn(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deviance between predictions and true values."""
        return 2 * (self._loss_fn(y_pred, y_true) - self._loss_fn(y_true, y_true))

    def forward(self):
        beta_tensor, intercept_tensor = self._get_params()
        return BaseGLM._get_loss(
            beta_tensor,
            intercept_tensor,
            self._loss_fn,
            self.dataloader,
            inv_link_fn=self.inv_link_fn,
            fit_intercept=self.fit_intercept,
        )

    def predict(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict with link function applied."""
        beta_tensor, intercept_tensor = self._get_params(beta_value)
        return BaseGLM._get_raw_prediction(
            beta_tensor,
            intercept_tensor,
            self.dataloader,
            X=X,
            inv_link_fn=self.inv_link_fn,
            fit_intercept=self.fit_intercept,
        )

    def loss(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss with inverse link applied."""
        beta_tensor, intercept_tensor = self._get_params(beta_value)
        return BaseGLM._get_loss(
            beta_tensor,
            intercept_tensor,
            self._loss_fn,
            self.dataloader,
            X=X,
            y=y,
            inv_link_fn=self.inv_link_fn,
            fit_intercept=self.fit_intercept,
        )

    def score(self, beta_value=None, X=None, y=None):
        y_model = self.predict(beta_value, X)
        y_true = self._get_target_values(X, y)
        y_null = _get_central_tendency(y_true, type="mean")
        model_deviance, null_deviance = (
            self.deviance_fn(y_model, y_true),
            self.deviance_fn(y_null, y_true),
        )
        return _get_regression_score(model_deviance, null_deviance)
