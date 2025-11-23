from abc import ABC, abstractmethod
from typing import Callable

import torch

from rlaopt.atoms import Atom
from rlaopt.data import DataLoader, Dataset
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.utils.device_utils import move_to_source_device

from .loss_factory import _get_loss_function
from .loss_types import LossType


class _BaseLinearModel(Atom, ABC):
    """Base class for all Linear Models."""

    def __init__(
        self,
        loss_type: LossType,
        beta: Variable,
        dataloader: DataLoader,
        fit_intercept: bool = True,
        **loss_kwargs,
    ):
        feature_dim = dataloader.dataset.feature_dimension
        if loss_type in [LossType.LOGISTIC, LossType.MULTINOMIAL]:
            target_dim = int(dataloader.dataset.y.max().item())
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

        # Add intercept if specified
        if fit_intercept:
            intercept_ = Variable(
                (target_dim,),
                name="intercept",
                device=beta_value.device,
                dtype=beta_value.dtype,
            )
            exprs["intercept"] = intercept_

        super().__init__(exprs, {}, None)

        _loss_fn = _get_loss_function(loss_type)
        if loss_type == LossType.POISSON_GAMMA:
            self._loss_fn = _loss_fn(loss_kwargs["power"], reduction="mean")
        else:
            self._loss_fn = _loss_fn(reduction="mean", **loss_kwargs)
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

    @abstractmethod
    def predict(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    def forward(self):
        beta_tensor, intercept_tensor = self._get_params()

        return _BaseLinearModel._get_loss(
            beta_tensor,
            intercept_tensor,
            self._loss_fn,
            self.dataloader,
            fit_intercept=self.fit_intercept,
        )

    def loss(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta_tensor, intercept_tensor = self._get_params(beta_value)

        return _BaseLinearModel._get_loss(
            beta_tensor,
            intercept_tensor,
            self._loss_fn,
            self.dataloader,
            X=X,
            y=y,
            fit_intercept=self.fit_intercept,
        )

    def is_smooth(self):
        return True

    def is_proxable(self):
        return False

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

    @staticmethod
    def _get_loss(
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        loss_fn: Callable,
        dataloader: DataLoader,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        inv_link_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fit_intercept: bool = True,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            beta: Model parameters tensor.
            intercept: Intercept tensor (can be None if fit_intercept=False).
            loss_fn: Loss function.
            dataloader: DataLoader containing dataset.
            X: Optional input features. If None, uses training dataset.
            y: Optional targets. Required if X is provided.
            inv_link_fn: Optional inverse link function for GLMs.
            fit_intercept: Whether model includes intercept.

        Returns:
            Loss value
        """
        # Case 1: Test data provided
        if _has_test_data(X, y):
            predictions = _BaseLinearModel._get_raw_prediction(
                beta,
                intercept,
                dataloader,
                X=X,
                inv_link_fn=inv_link_fn,
                fit_intercept=fit_intercept,
            )
            return loss_fn(predictions, y)

        # Case 2: Training data - in-memory Dataset
        if isinstance(dataloader.dataset, Dataset):
            predictions = _BaseLinearModel._get_raw_prediction(
                beta,
                intercept,
                dataloader,
                inv_link_fn=inv_link_fn,
                fit_intercept=fit_intercept,
            )
            return loss_fn(predictions, dataloader.dataset.y)

        # Case 3: Training data - BatchedDataset

        total_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in dataloader:
            loss = _BaseLinearModel._batch_loss(
                beta,
                intercept,
                X_batch,
                y_batch,
                loss_fn,
                normalize=False,
                inv_link_fn=inv_link_fn,
                fit_intercept=fit_intercept,
            )
            total_loss += loss
            n_samples += X_batch.shape[0]

        return total_loss / n_samples

    @staticmethod
    def _get_raw_prediction(
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        dataloader: DataLoader,
        X: torch.Tensor | None = None,
        inv_link_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fit_intercept: bool = True,
    ) -> torch.Tensor:
        """Generate predictions using the model parameters.

        Args:
            beta: Model parameters tensor.
            intercept: Intercept tensor (can be None if fit_intercept=False).
            dataloader: DataLoader containing the dataset.
            X: Optional input data for inference. If None, uses training dataset.
            inv_link_fn: Optional inverse link function to apply to linear predictor.
            fit_intercept: Whether model includes intercept.

        Returns:
            Predictions as a torch.Tensor with gradients preserved for backprop.
        """
        # Case 1: Inference on new data
        if X is not None:
            move_to_source_device(X, beta)
            linear_pred = X @ beta
            if fit_intercept:
                linear_pred = linear_pred + intercept
            return inv_link_fn(linear_pred) if inv_link_fn else linear_pred

        # Case 2: Training dataset - fast path for full in-memory
        if isinstance(dataloader.dataset, Dataset):
            linear_pred = dataloader.dataset.X @ beta
            if fit_intercept:
                linear_pred = linear_pred + intercept
            return inv_link_fn(linear_pred) if inv_link_fn else linear_pred

        # Case 3: Training dataset - batched (for both Dataset and BatchedDataset)
        predictions = []
        for X_batch, _ in dataloader:
            batch_pred, _ = _BaseLinearModel._batch_raw_predict(
                beta, intercept, X_batch, fit_intercept=fit_intercept
            )
            if inv_link_fn:
                batch_pred = inv_link_fn(batch_pred)
            predictions.append(batch_pred)
        return torch.cat(predictions, dim=0)

    @staticmethod
    def _batch_loss(
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        loss_fn: Callable,
        normalize: bool = True,
        inv_link_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fit_intercept: bool = True,
    ) -> torch.Tensor:
        """Compute loss on a single batch of data.

        Args:
            beta: Model parameters tensor
            intercept: Intercept tensor (can be None if fit_intercept=False)
            X_batch: input features for the batch
            y_batch: target values for the batch
            loss_fn: loss function to use
            normalize: if True, normalize loss by batch size
            inv_link_fn: Optional inverse link function to apply to linear predictor
            fit_intercept: Whether model includes intercept

        Returns:
            Loss for the batch
        """
        pred_batch, y_batch = _BaseLinearModel._batch_raw_predict(
            beta,
            intercept,
            X_batch,
            y_batch,
            inv_link_fn=inv_link_fn,
            fit_intercept=fit_intercept,
        )

        # Compute loss with sum reduction for proper averaging
        if normalize:
            batch_loss = loss_fn(pred_batch, y_batch)
        else:
            # Temporarily set reduction to 'sum' to get total loss for the batch
            original_reduction = _change_loss_reduction(loss_fn, "sum")
            batch_loss = loss_fn(pred_batch, y_batch)
            _change_loss_reduction(loss_fn, original_reduction)
        return batch_loss

    @staticmethod
    def _batch_raw_predict(
        beta: torch.Tensor,
        intercept: torch.Tensor | None,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor | None = None,
        inv_link_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        fit_intercept: bool = True,
    ) -> torch.Tensor:
        """Predict on a single batch of data.

        Args:
            beta: Model parameters tensor
            intercept: Intercept tensor (can be None if fit_intercept=False)
            X_batch: input features for the batch
            y_batch: target values for the batch (optional)
            inv_link_fn: Optional inverse link function to apply to linear predictor
            fit_intercept: Whether model includes intercept

        Returns:
            Predictions for the batch
        """
        if y_batch is not None:
            X_batch, y_batch = move_to_source_device((X_batch, y_batch), beta)
        else:
            X_batch = move_to_source_device(X_batch, beta)

        linear_pred = X_batch @ beta
        if fit_intercept:
            linear_pred = linear_pred + intercept

        if inv_link_fn:
            return inv_link_fn(linear_pred), y_batch
        else:
            return linear_pred, y_batch


def _change_loss_reduction(loss_fn: torch.nn.modules.loss._Loss, reduction: str) -> str:
    """Change the reduction method of a loss function."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction type: {reduction}")
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
