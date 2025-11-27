"""Private inverse link function module."""

from abc import ABC, abstractmethod
from enum import Enum

import torch


class InverseLinkFunction(ABC):
    """Abstract base class for inverse link functions in generalized linear models.

    In GLM terminology, the link function g relates the mean parameter μ to the
    linear predictor η: g(μ) = η. The inverse link function g⁻¹ maps from the
    linear predictor back to the mean parameter: μ = g⁻¹(η).

    Notation:
    - Linear predictor: η = Xβ
    - Mean parameter: μ = g⁻¹(η)

    Subclasses must implement __call__() to compute μ from η.
    """

    @abstractmethod
    def __call__(self, linear_pred: torch.Tensor) -> torch.Tensor:
        """Apply the inverse link function to transform linear predictor to mean parameter.

        Args:
            linear_pred: Linear predictor η = Xβ

        Returns:
            Mean parameter μ = g⁻¹(η)
        """
        pass


class IdentityInverseLink(InverseLinkFunction):
    """Identity inverse link function: g⁻¹(η) = η.

    Corresponds to the identity link g(μ) = μ.

    Commonly used for:
    - Linear regression (Gaussian family)
    - Models where the response is already on an unconstrained scale
    """

    def __call__(self, linear_pred):
        """Return linear predictor unchanged.

        Args:
            linear_pred: Linear predictor η

        Returns:
            μ = η (identity transformation)
        """
        return linear_pred


class ExpInverseLink(InverseLinkFunction):
    """Exponential inverse link function: g⁻¹(η) = exp(η).

    Corresponds to the log link g(μ) = log(μ).

    Commonly used for:
    - Poisson regression (count data)
    - Models with positive continuous responses
    - When multiplicative effects are expected
    """

    def __call__(self, linear_pred):
        """Apply exponential function to map linear predictor to positive reals.

        Args:
            linear_pred: Linear predictor η (unconstrained)

        Returns:
            μ = exp(η) > 0
        """
        return torch.exp(linear_pred)


class InverseLinkType(Enum):
    """Enumeration of available inverse link function types.

    Attributes:
        IDENTITY: Identity g⁻¹(η) = η
        EXP: Exponential g⁻¹(η) = exp(η)
    """

    IDENTITY = "identity"
    EXP = "exp"


INVERSE_LINK_FNS = {
    InverseLinkType.IDENTITY: IdentityInverseLink,
    InverseLinkType.EXP: ExpInverseLink,
}


def get_inverse_link_function(
    inverse_link_type: InverseLinkType,
) -> InverseLinkFunction:
    """Factory function to instantiate an inverse link function from its type.

    Args:
        inverse_link_type: InverseLinkType enum specifying which function to create

    Returns:
        Instantiated InverseLinkFunction object

    Example:
        >>> inv_link = _get_inverse_link_function(InverseLinkType.SIGMOID)
        >>> probs = inv_link(logits)
    """
    return INVERSE_LINK_FNS[inverse_link_type]()
