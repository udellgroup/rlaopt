from enum import Enum


class LossType(Enum):
    """Enumeration of different loss types for GLM models."""

    GAMMA = "gamma"
    HUBER = "huber"
    INV_GAUSS = "inverse_gaussian"
    L1_LOSS = "l1_loss"
    LEAST_SQUARES = "least_squares"
    LOGISTIC = "logistic"
    MULTINOMIAL = "multinomial"
    POISSON = "poisson"
    POISSON_GAMMA = "poisson_gamma"
