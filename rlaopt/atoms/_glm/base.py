from abc import ABC, abstractmethod

import torch

from rlaopt.atoms.atom_expression import AtomExpression
from rlaopt.expression.expression import Variable
from rlaopt.dataloader import DataLoader
from rlaopt._typing import TensorDict

from . import core
from .loss_factory import get_loss_function
from .loss_types import LossType


class _BaseGLM(AtomExpression, ABC):
    """
    Base class for Generalized Linear Models (GLMs).
    """

    def __init__(
        self, dataloader: DataLoader, beta: Variable, loss_type: LossType, **loss_kwargs
    ):
        super().__init__()
        self.register_variable(beta)
        self._loss_fn = get_loss_function(loss_type)(reduction="mean", **loss_kwargs)
        self.dataloader = dataloader

        self._validate()

    def _validate(self):
        """Validate dimension compatibility between dataset and variable."""
        expected_features = self.dataloader.dataset.feature_dimension
        beta = self.get_variable(self.var_name)
        actual_features = beta.data.shape[0]

        if expected_features != actual_features:
            raise ValueError(
                f"Dimension mismatch: dataset features {expected_features} "
                f"and variable size {actual_features} do not match."
            )

    @abstractmethod
    def score(self, params: TensorDict):
        pass

    def predict(
        self,
        beta: torch.nn.Parameter | torch.Tensor | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if beta is None:
            beta = self.get_variable(self.var_name)
        return core.compute_prediction(beta, self.dataloader, X)

    def loss(
        self,
        beta: torch.nn.Parameter | torch.Tensor | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if beta is None:
            beta = self.get_variable(self.var_name)
        return core.compute_loss(beta, self._loss_fn, self.dataloader, X=X, y=y)

    def forward(self):
        return core.compute_loss(
            self.get_variable(self.var_name),
            self._loss_fn,
            self.dataloader,
            X=None,
            y=None,
        )

    def is_smooth(self):
        return True

    def is_subsamplable(self):
        return False

    def is_proxable(self):
        return False

    def subsample(self, indices):
        raise NotImplementedError("Subsampling not supported for GLM atoms.")

    def to_cvxpy(self):
        raise NotImplementedError("Conversion to CVXPY not implemented for GLM atoms.")


class _GLMClassifier(_BaseGLM, ABC):
    """
    GLM for classification tasks.
    """

    def __init__(self, dataloader, beta, loss_type, **loss_kwargs):
        if loss_type not in {LossType.LOGISTIC, LossType.MULTINOMIAL}:
            raise ValueError(
                "Invalid loss type for classification. Use LOGISTIC or MULTINOMIAL."
            )
        super().__init__(dataloader, beta, loss_type, **loss_kwargs)

    @abstractmethod
    def predict_proba(
        self, params: TensorDict, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            params: Model parameters
            X: Optional input data. If None, uses training dataset.

        Returns:
            Class probabilities
        """
        # Default implementation - subclasses can override
        raise NotImplementedError("Subclasses must implement predict_proba")
