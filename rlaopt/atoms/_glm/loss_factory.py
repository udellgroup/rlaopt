import torch
from .loss_types import LossType

LOSSES = {
    LossType.HUBER: torch.nn.HuberLoss,
    LossType.L1_LOSS: torch.nn.L1Loss,
    LossType.LEAST_SQUARES: torch.nn.MSELoss,
    LossType.LOGISTIC: torch.nn.BCEWithLogitsLoss,
    LossType.MULTINOMIAL: torch.nn.CrossEntropyLoss,
    LossType.POISSON: torch.nn.PoissonNLLLoss,
}


def get_loss_function(loss_type: LossType):
    """Factory function to get the appropriate loss function based on the loss type.

    Args:
        loss_type (LossType): The type of loss function to retrieve.

    Returns:
        An instance of the specified loss function.

    Raises:
        ValueError: If the provided loss type is not recognized.
    """
    if loss_type not in LOSSES:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return LOSSES[loss_type]
