"""This module defines configuration classes for various preconditioners.

It includes abstract base classes and specific configuration classes for different
preconditioner types, as well as validation utilities.
"""

from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any, Union

from .enums import DampingMode
from rlaopt.utils import (
    _is_str,
    _is_nonneg_float,
    _is_pos_int,
    _is_sketch,
)


__all__ = [
    "PreconditionerConfig",
    "IdentityConfig",
    "NewtonConfig",
    "NystromConfig",
    "SkPreConfig",
    "_is_precond_config",
]


@dataclass(kw_only=True, frozen=False)
class PreconditionerConfig(ABC):
    """Abstract base class for preconditioner configurations.

    This class serves as a base for all specific preconditioner configurations.
    """

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return asdict(self)


@dataclass(kw_only=True, frozen=False)
class IdentityConfig(PreconditionerConfig):
    """Configuration for the Identity preconditioner.

    This configuration doesn't require any specific parameters.
    """

    pass


@dataclass(kw_only=True, frozen=False)
class NewtonConfig(PreconditionerConfig):
    """Configuration for the Newton preconditioner.

    Attributes:
        rho (float): Damping parameter for the Newton method.
    """

    rho: float

    def __post_init__(self):
        """Validate the configuration parameters after initialization."""
        _is_nonneg_float(self.rho, "rho")


@dataclass(kw_only=True, frozen=False)
class NystromConfig(PreconditionerConfig):
    """Configuration for the Nyström preconditioner.

    Attributes:
        rank (int): Rank of the Nyström approximation.
        rho (float): Regularization parameter.
        sketch (str): Type of sketching method to use. Defaults to "ortho".
        damping_mode (DampingMode or str): Damping mode to use. Can be specified as
            DampingMode.ADAPTIVE, DampingMode.NON_ADAPTIVE,
            "adaptive", or "non_adaptive". Defaults to DampingMode.ADAPTIVE.
    """

    rank: int
    rho: float
    sketch: str = "ortho"
    damping_mode: Union[DampingMode, str] = DampingMode.ADAPTIVE

    def __post_init__(self):
        """Validate the configuration parameters after initialization."""
        _is_pos_int(self.rank, "rank")
        _is_nonneg_float(self.rho, "rho")

        _is_str(self.sketch, "sketch")
        _is_sketch(self.sketch)
        self.damping_mode = DampingMode._from_str(self.damping_mode, "damping_mode")


@dataclass(kw_only=True, frozen=False)
class SkPreConfig(PreconditionerConfig):
    """Configuration for the Sketched Preconditioner (SkPre).

    Attributes:
        sketch_size (int): Size of the sketch.
        rho (float): Regularization parameter.
        sketch (str): Type of sketching method to use. Defaults to "sparse".
    """

    sketch_size: int
    rho: float
    sketch: str = "sparse"

    def __post_init__(self):
        """Validate the configuration parameters after initialization."""
        _is_pos_int(self.sketch_size, "sketch_size")
        _is_nonneg_float(self.rho, "rho")
        _is_str(self.sketch, "sketch")
        _is_sketch(self.sketch)


def _is_precond_config(param: Any, param_name: str):
    """Check if a parameter is an instance of PreconditionerConfig.

    Args:
        param (Any): The parameter to check.
        param_name (str): The name of the parameter (for error reporting).

    Raises:
        TypeError: If the parameter is not an instance of PreconditionerConfig.

    Example:
        >>> config = NewtonConfig(rho=1e-4)
        >>> _is_precond_config(config, "my_config")  # No exception raised
        >>> _is_precond_config("not a config", "bad_config")  # Raises TypeError
    """
    if not isinstance(param, PreconditionerConfig):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type PreconditionerConfig"
        )
