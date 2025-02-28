from rlaopt.preconditioners.identity import Identity
from rlaopt.preconditioners.newton import Newton
from rlaopt.preconditioners.nystrom import Nystrom
from rlaopt.preconditioners.skpre import SkPre

from rlaopt.preconditioners.configs import (
    PreconditionerConfig,
    IdentityConfig,
    NewtonConfig,
    NystromConfig,
    SkPreConfig,
)

CONFIG_TO_PRECONDITIONER = {
    IdentityConfig: Identity,
    NewtonConfig: Newton,
    NystromConfig: Nystrom,
    SkPreConfig: SkPre,
}


def _get_precond(precond_config: PreconditionerConfig):
    # Get the class of precond_config
    precond_config_class = precond_config.__class__
    # Get the corresponding preconditioner class
    precond_class = CONFIG_TO_PRECONDITIONER.get(precond_config_class)

    return precond_class(precond_config)
