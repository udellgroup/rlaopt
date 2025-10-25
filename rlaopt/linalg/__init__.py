"""Module for linear algebra."""

from rlaopt.linalg.lin_sys import LinSys as LinSys
from rlaopt.linalg.preconditioners import IdentityConfig as IdentityConfig
from rlaopt.linalg.preconditioners import NystromConfig as NystromConfig
from rlaopt.linalg.preconditioners import Preconditioner as Preconditioner
from rlaopt.linalg.preconditioners import PreconditionerConfig as PreconditionerConfig
from rlaopt.linalg.preconditioners import get_preconditioner as get_preconditioner
