"""Tweedie loss functions.

This module provides loss functions for GLMs with link functions,
from the Tweedie family of distributions.
"""

import torch
from torch.nn.modules.loss import _Loss


class TweedieLoss(_Loss):
    """Tweedie loss for regression.

    The Tweedie distribution is a family of distributions that includes:
    - Poisson (power=1)
    - Compound Poisson-Gamma (1 < power < 2)
    - Gamma (power=2)
    - Inverse Gaussian (power=3)p

    The Tweedie loss is defined as:
        L(y, ŷ) = -[y * ŷ^(1-p) / (1-p) - ŷ^(2-p) / (2-p)]
                  (ignoring y-dependent constants)

    For p=1 (Poisson), this reduces to:
        L(y, ŷ) = ŷ - y * log(ŷ)

    For p=2 (Gamma), this reduces to:
        L(y, ŷ) = y/ŷ + log(ŷ)

    For p=3 (Inverse Gaussian), this reduces to:
        L(y, ŷ) = (y - ŷ)² / (2 * y * ŷ²)

    Args:
        power: Tweedie power parameter (p).
               Common values: 1.0 (Poisson), 1.5 (Compound), 2.0 (Gamma),
                              3.0 (Inverse Gaussian).
               Also accepts values in range (1, 2) for compound distributions.
               Default: 1.5
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'

    Shape:
        - Input: (N, *) where * means any number of additional dimensions
        - Target: (N, *), same shape as input
        - Output: scalar if reduction is 'mean' or 'sum', otherwise (N, *)

    Note:
        Predictions (input) must be positive.
        Target values must be non-negative.
    """

    def __init__(self, power: float = 1.5, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        # Allow common discrete values or continuous range (1, 2)
        valid_discrete = {1.0, 1.5, 2.0, 3.0}
        if power not in valid_discrete and not (1.0 < power < 2.0):
            raise ValueError(
                f"Power must be one of {valid_discrete} or in range (1, 2). "
                f"Got {power}. Common values: 1.0 (Poisson), "
                f"1.5 (Compound Poisson-Gamma), "
                f"2.0 (Gamma), 3.0 (Inverse Gaussian)"
            )
        self.power = power

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Tweedie loss.

        Args:
            input_: Predicted values (must be positive)
            target: Ground truth values (must be non-negative)

        Returns:
            Loss value
        """
        return tweedie_loss(input_, target, power=self.power, reduction=self.reduction)


def tweedie_loss(
    input_: torch.Tensor,
    target: torch.Tensor,
    power: float = 1.5,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Functional form of Tweedie loss (negative log-likelihood).

    Args:
        input_: Predicted values (ŷ), shape (N, *)
        target: Target values (y), shape (N, *)
        power: Tweedie power parameter
        reduction: 'none' | 'mean' | 'sum'
        eps: Small constant for numerical stability

    Returns:
        Loss tensor (negative log-likelihood, ignoring constants)
    """
    # Input validation
    if torch.any(target < 0):
        raise ValueError("Target values must be non-negative")

    # Compute negative log-likelihood based on power
    if power == 1:
        # Poisson distribution
        # NLL = ŷ - y*log(ŷ)  [ignoring log(y!) constant]
        loss = input_ - target * torch.log(input_ + eps)

    elif power == 2:
        # Gamma distribution
        # NLL = log(ŷ) + y/ŷ  [ignoring log(y) and other constants]
        loss = torch.log(input_ + eps) + target / (input_ + eps)

    elif power == 3:
        # Inverse Gaussian distribution
        # NLL = (y - ŷ)² / (2 * ŷ² * y)  [ignoring constants not depending on ŷ]
        loss = torch.pow(target - input_, 2) / (2 * torch.pow(input_, 2) * target + eps)

    else:
        # General Tweedie case (mostly for 1 < p < 2)
        # NLL = -[y * ŷ^(1-p) / (1-p) - ŷ^(2-p) / (2-p)]
        # (ignoring y-dependent constants)
        p1 = 1 - power
        p2 = 2 - power

        # First term: y * ŷ^(1-p) / (1-p)
        term1 = target * torch.pow(input_ + eps, p1) / p1

        # Second term: ŷ^(2-p) / (2-p)
        term2 = torch.pow(input_ + eps, p2) / p2

        loss = -term1 + term2

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()


class PoissonLoss(TweedieLoss):
    """Poisson loss for count data (Tweedie with power=1).

    The Poisson distribution is used for modeling count data (non-negative integers).
    The deviance loss is:
        L(y, ŷ) = 2 * [y * log(y/ŷ) - (y - ŷ)]

    Common applications:
        - Number of events in a fixed time period
        - Count data (claims, calls, visits, etc.)
        - Rare events

    Args:
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'

    Note:
        This is equivalent to torch.nn.PoissonNLLLoss but uses the Tweedie
        parameterization. Predictions and targets must be non-negative.
        For targets, zeros are handled correctly.
    """

    def __init__(self, reduction: str = "mean"):
        # Poisson is Tweedie with power=1
        super().__init__(power=1.0, reduction=reduction)


class GammaLoss(TweedieLoss):
    """Gamma loss for positive continuous data (Tweedie with power=2).

    The Gamma distribution is used for modeling strictly positive continuous data.
    The deviance loss is:
        L(y, ŷ) = log(ŷ/y) + y/ŷ

    Common applications:
        - Claim severity (insurance)
        - Survival/duration times
        - Waiting times
        - Positive continuous measurements

    Args:
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'

    Note:
        Both predictions and targets must be strictly positive.
    """

    def __init__(self, reduction: str = "mean"):
        # Gamma is Tweedie with power=2
        super().__init__(power=2.0, reduction=reduction)


class InverseGaussianLoss(TweedieLoss):
    """Inverse Gaussian loss for positive continuous data (Tweedie with power=3).

    The Inverse Gaussian (Wald) distribution is used for modeling positive
    continuous data, particularly when data exhibits right skewness.
    The deviance loss is:
        L(y, ŷ) = (y - ŷ)² / (y * ŷ²)

    Common applications:
        - Lifetime/degradation data
        - Response times
        - First passage times
        - Financial data (stock prices, returns)

    Args:
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'

    Note:
        Both predictions and targets must be strictly positive.
        The Inverse Gaussian has heavier right tail than Gamma.
    """

    def __init__(self, reduction: str = "mean"):
        # Inverse Gaussian is Tweedie with power=3
        super().__init__(power=3.0, reduction=reduction)


class CompoundPoissonGammaLoss(TweedieLoss):
    """Compound Poisson-Gamma loss (Tweedie with power in (1,2)).

    The Compound Poisson-Gamma distribution models data that has:
    1. A probability mass at zero (no events)
    2. Positive continuous values when events occur

    This is the default Tweedie loss and arises in many applications.

    Common applications:
        - Insurance claim amounts (many policyholders have zero claims)
        - Rainfall data (many days with no rain, positive amounts when it rains)
        - Customer spending (many customers spend nothing, others spend varying amounts)
        - Healthcare costs (many people have zero costs, others have positive costs)

    Args:
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'

    Note:
        Predictions must be positive, targets can be zero or positive.
        Power=1.5 is a common default, but any value in (1, 2) is valid.
    """

    def __init__(self, power: float, reduction: str = "mean"):
        super().__init__(power=power, reduction=reduction)
