from enum import Enum


class LossType(Enum):
    """Enumeration of different loss types for GLM models."""

    HUBER = "huber"
    L1_LOSS = "l1_loss"
    LEAST_SQUARES = "least_squares"
    LOGISTIC = "logistic"
    MULTINOMIAL = "multinomial"
    POISSON = "poisson"
