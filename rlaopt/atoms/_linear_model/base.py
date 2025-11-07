from abc import ABC, abstractmethod
from typing import Callable

import torch

from rlaopt.atoms import AtomExpression
from rlaopt.expression import Variable
from rlaopt.ext_tensordict import TensorDict
from rlaopt.data import DataLoader, Dataset
from rlaopt.utils.device_utils import move_to_source_device

from .loss_factory import get_loss_function
from .loss_types import LossType


class _BaseLinearModel(AtomExpression, ABC):
    """
    Base class for Linear Models.
    """

    def __init__(
        self, dataloader: DataLoader, beta: Variable, loss_type: LossType, **loss_kwargs
    ):
        super().__init__()
        self.register_input(beta)
        self._loss_fn = get_loss_function(loss_type)(reduction="mean", **loss_kwargs)
        self.dataloader = dataloader

        self._validate()

    def _validate(self):
        """Validate dimension compatibility between dataset and variable."""
        expected_features = self.dataloader.dataset.feature_dimension
        beta = self.get_input().forward()
        actual_features = beta.data.shape[0]

        if expected_features != actual_features:
            raise ValueError(
                f"Dimension mismatch: dataset features {expected_features} "
                f"and variable size {actual_features} do not match."
            )

    @abstractmethod
    def score(
        self, 
        beta_value: TensorDict | None = None, 
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None ) -> float:
        pass

    def predict(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta_tensor = _get_beta(self, beta_value)
        return compute_prediction(beta_tensor, self.dataloader, X)

    def loss(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        beta_tensor= _get_beta(self, beta_value)
        return compute_loss(beta_tensor, self._loss_fn, self.dataloader, X=X, y=y)

    def forward(self):
        return self.loss()

    def is_smooth(self):
        return True

    def is_subsamplable(self):
        return False

    def is_proxable(self):
        return False

    def subsample(self, indices):
        raise NotImplementedError("Subsampling not supported for GLM atoms.")
    

class _LinearClassifier(_BaseLinearModel, ABC):
    """
    Linear model for classification tasks.
    """

    def __init__(self, dataloader, beta, loss_type, **loss_kwargs):
        if loss_type not in {LossType.LOGISTIC, LossType.MULTINOMIAL}:
            raise ValueError(
                "Invalid loss type for classification. Use LOGISTIC or MULTINOMIAL."
            )
        super().__init__(dataloader, beta, loss_type, **loss_kwargs)

    @abstractmethod
    def predict_proba(
        self, beta_value: TensorDict, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Class probabilities
        """
        # Default implementation - subclasses can override
        raise NotImplementedError("Subclasses must implement predict_proba")
    
    @abstractmethod
    def predict_classes(
        self, beta_value: TensorDict, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            beta: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Predicted class labels
        """
        # Default implementation - subclasses can override
        raise NotImplementedError("Subclasses must implement predict_classes")

def compute_loss(
    beta: torch.Tensor,
    loss_fn: Callable,
    dataloader: DataLoader,
    X: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute loss.

    Args:
        beta: Model parameters tensor.
        X: Optional input features. If None, uses training dataset.
        y: Optional targets. Required if X is provided.

    Returns:
        Loss value
    """
    # Case 1: Test data provided
    if _has_test_data(X, y):
        predictions = compute_prediction(beta, dataloader, X=X)
        return loss_fn(predictions, y)

    # Case 2: Training data - in-memory Dataset
    if isinstance(dataloader.dataset, Dataset):
        predictions = compute_prediction(beta, dataloader)
        return loss_fn(predictions, dataloader.dataset.y)

    # Case 3: Training data - BatchedDataset
    if dataloader is None:
        raise ValueError("Dataloader required for batched prediction.")

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        loss = batch_loss(beta, X_batch, y_batch, loss_fn, normalize=False)
        total_loss += loss
        n_samples += X_batch.shape[0]

    return total_loss / n_samples


def compute_prediction(
    beta: torch.Tensor,
    dataloader: DataLoader,
    X: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate predictions using the model parameters.

    Args:
        beta: Model parameters tensor.
        X: Optional input data for inference. If None, uses training dataset.
        use_full_dataset: If True and using training dataset, compute on full
                        dataset at once (only for in-memory Dataset). If False,
                        uses batching. Ignored when X is provided.

    Returns:
        Predictions as a torch.Tensor with gradients preserved for backprop.
    """

    # Case 1: Inference on new data
    if X is not None:
        move_to_source_device(X, beta)
        return X @ beta

    # Case 2: Training dataset - fast path for full in-memory
    if isinstance(dataloader.dataset, Dataset):
        return dataloader.dataset.X @ beta

    # Case 3: Training dataset - batched (for both Dataset and BatchedDataset)
    predictions = []
    if dataloader is None:
        raise ValueError("dataloader required for batched prediction")
    for X_batch, _ in dataloader:
        batch_pred, _ = batch_predict(beta, X_batch)
        predictions.append(batch_pred)
    return torch.cat(predictions, dim=0)


def batch_loss(
    beta: torch.Tensor,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    loss_fn: Callable,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute loss on a single batch of data.

    Args:
        beta: Model parameters tensor
        X_batch: input features for the batch
        y_batch: target values for the batch
        loss_fn: loss function to use
        normalize: if True, normalize loss by batch size

    Returns:
        Loss for the batch
    """
    pred_batch, y_batch = batch_predict(beta, X_batch, y_batch)
    # Compute loss with sum reduction for proper averaging
    if normalize:
        batch_loss = loss_fn(pred_batch, y_batch)
    else:
        # Temporarily set reduction to 'sum' to get total loss for the batch
        original_reduction = _change_loss_reduction(loss_fn, "sum")
        batch_loss = loss_fn(pred_batch, y_batch)
        _change_loss_reduction(loss_fn, original_reduction)
    return batch_loss


def batch_predict(
    beta: torch.Tensor,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Predict on a single batch of data.

    Args:
        beta: Model parameters tensor
        X_batch: input features for the batch
        y_batch: target values for the batch (optional)

    Returns:
        Predictions for the batch
    """
    if y_batch is not None:
        X_batch, y_batch = move_to_source_device((X_batch, y_batch), beta)
    else:
        X_batch = move_to_source_device(X_batch, beta)
    return X_batch @ beta, y_batch


def _change_loss_reduction(loss_fn: torch.nn.modules.loss._Loss, reduction: str) -> str:
    """Change the reduction method of a loss function."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    original_reduction = loss_fn.reduction
    loss_fn.reduction = reduction
    return original_reduction


def _get_beta(glm: _BaseLinearModel, beta: torch.Tensor | None = None)-> torch.Tensor:
    if beta:
        return beta['beta']
    else:
        return glm.get_input().forward()
    

def _has_test_data(X: torch.Tensor | None, y: torch.Tensor | None) -> bool:
    if X is not None:
        if y is None:
            raise ValueError("Must provide y when X is specified")
        else:
            return True
    else:
        return False