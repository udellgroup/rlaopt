import torch
from torch.nn.modules.loss import _Loss

from .custom_losses.tweedie import (
    CompoundPoissonGammaLoss,
    GammaLoss,
    InverseGaussianLoss,
    PoissonLoss,
)
from .loss_types import LossType

LOSSES = {
    LossType.POISSON_GAMMA: CompoundPoissonGammaLoss,
    LossType.GAMMA: GammaLoss,
    LossType.HUBER: torch.nn.HuberLoss,
    LossType.INV_GAUSS: InverseGaussianLoss,
    LossType.L1_LOSS: torch.nn.L1Loss,
    LossType.LEAST_SQUARES: torch.nn.MSELoss,
    LossType.LOGISTIC: torch.nn.BCEWithLogitsLoss,
    LossType.MULTINOMIAL: torch.nn.CrossEntropyLoss,
    LossType.POISSON: PoissonLoss,
}


def get_loss_function(loss_type: LossType) -> _Loss:
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
