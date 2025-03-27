"""This module provides a factory for creating preconditioners.

It includes mappings between preconditioner configurations and their corresponding
preconditioner classes, as well as a function to instantiate preconditioners.
"""

from .identity import Identity
from .newton import Newton
from .nystrom import Nystrom
from .skpre import SkPre

from .configs import (
    PreconditionerConfig,
    IdentityConfig,
    NewtonConfig,
    NystromConfig,
    SkPreConfig,
)


# Mapping of configuration classes to their corresponding preconditioner classes
CONFIG_TO_PRECONDITIONER = {
    IdentityConfig: Identity,
    NewtonConfig: Newton,
    NystromConfig: Nystrom,
    SkPreConfig: SkPre,
}


__all__ = ["_get_precond"]


def _get_precond(precond_config: PreconditionerConfig):
    """Create and return a preconditioner instance based on the given configuration.

    This function acts as a factory, instantiating the appropriate preconditioner
    class based on the type of the provided configuration object.

    Args:
        precond_config (PreconditionerConfig): The configuration object for
        the desired preconditioner.

    Returns:
        Preconditioner: An instance of the appropriate preconditioner class.

    Raises:
        KeyError: If no corresponding preconditioner class
        is found for the given configuration.

    Example:
        >>> from rlaopt.preconditioners.configs import NystromConfig
        >>> config = NystromConfig(rank=100, sketch='gaussian', rho=1e-4)
        >>> preconditioner = _get_precond(config)
        >>> print(type(preconditioner))
        <class 'rlaopt.preconditioners.nystrom.Nystrom'>
    """
    # Get the class of precond_config
    precond_config_class = precond_config.__class__
    # Get the corresponding preconditioner class
    precond_class = CONFIG_TO_PRECONDITIONER.get(precond_config_class)

    if precond_class is None:
        raise KeyError(
            f"No preconditioner found for configuration: {precond_config_class}"
        )

    return precond_class(precond_config)
