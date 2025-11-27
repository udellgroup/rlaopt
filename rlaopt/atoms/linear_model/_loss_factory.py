from enum import Enum

import torch
from torch.nn.modules.loss import _Loss

from ._tweedie import (
    CompoundPoissonGammaLoss,
    GammaLoss,
    InverseGaussianLoss,
    PoissonLoss,
)


class LossType(Enum):
    """Enumeration of different loss types for GLM models."""

    GAMMA = "gamma"
    HUBER = "huber"
    INV_GAUSS = "inverse_gaussian"
    LEAST_SQUARES = "least_squares"
    LOGISTIC = "logistic"
    MULTINOMIAL = "multinomial"
    POISSON = "poisson"
    POISSON_GAMMA = "poisson_gamma"


LOSSES = {
    LossType.POISSON_GAMMA: CompoundPoissonGammaLoss,
    LossType.GAMMA: GammaLoss,
    LossType.HUBER: torch.nn.HuberLoss,
    LossType.INV_GAUSS: InverseGaussianLoss,
    LossType.LEAST_SQUARES: torch.nn.MSELoss,
    LossType.LOGISTIC: torch.nn.BCEWithLogitsLoss,
    LossType.MULTINOMIAL: torch.nn.CrossEntropyLoss,
    LossType.POISSON: PoissonLoss,
}


def get_loss_function(loss_type: LossType) -> _Loss:
    """Internal Factory function to get the appropriate loss function based on the loss type.

    Args:
        loss_type (LossType): The type of loss function to retrieve.

    Returns:
        An instance of the specified loss function.

    """
    return LOSSES[loss_type]
