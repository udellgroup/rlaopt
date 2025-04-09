# Import torch to ensure its libraries are linked properly for the extensions
import torch  # noqa: F401

# Now, import C++ and CUDA extensions
from . import _C  # noqa: F401
