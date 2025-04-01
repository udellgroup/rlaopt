"""Preconditioners module __init__.py file."""
from .configs import *
from .factory import *
from .preconditioner import *

# Collect __all__ from imported modules
__all__ = []
for module in [configs, factory, preconditioner]:
    if hasattr(module, "__all__"):
        __all__.extend(module.__all__)
