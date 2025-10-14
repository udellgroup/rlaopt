from typing import Callable

from rlaopt.datasets import Dataset
from rlaopt.dataloader import DataLoader
from rlaopt.utils.device_utils import move_to_source_device

import torch


def compute_loss(
    beta: torch.nn.Parameter | torch.Tensor,
    loss_fn: Callable,
    dataloader: DataLoader,
    X: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute loss.

    Args:
        beta: tensor storing value of model parameters beta
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
        raise ValueError("dataloader required for batched prediction")

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        loss = batch_loss(beta, X_batch, y_batch, loss_fn, normalize=False)
        total_loss += loss
        n_samples += X_batch.shape[0]

    return total_loss / n_samples


def compute_prediction(
    beta: torch.nn.Parameter | torch.Tensor,
    dataloader: DataLoader,
    X: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate predictions using the model parameters.

    Args:
        beta: tensor storing value of model parameters beta
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
    beta: torch.nn.Parameter | torch.Tensor,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    loss_fn: Callable,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute loss on a single batch of data.

    Args:
        beta: model parameters beta
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
        original_reduction = loss_fn.reduction
        loss_fn.reduction = "sum"
        batch_loss = loss_fn(pred_batch, y_batch)
        loss_fn.reduction = original_reduction
    return batch_loss


def batch_predict(
    beta: torch.nn.Parameter | torch.Tensor,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Predict on a single batch of data.

    Args:
        beta: model parameters beta
        X_batch: input features for the batch

    Returns:
        Predictions for the batch
    """
    if y_batch is not None:
        X_batch, y_batch = move_to_source_device((X_batch, y_batch), beta)
    else:
        X_batch = move_to_source_device(X_batch, beta)
    return X_batch @ beta, y_batch


def _has_test_data(X: torch.Tensor | None, y: torch.Tensor | None) -> bool:
    if X is not None:
        if y is None:
            raise ValueError("Must provide y when X is specified")
        else:
            return True
    else:
        return False
