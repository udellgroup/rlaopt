"""Linops module __init__.py file."""
from .distributed import *
from .mixins import *
from .simple import *
from .types import *

# Collect __all__ from imported modules
__all__ = []
for module in [distributed, mixins, simple, types]:
    if hasattr(module, "__all__"):
        __all__.extend(module.__all__)
