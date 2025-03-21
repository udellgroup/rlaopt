from typing import Any, Union, TYPE_CHECKING

import torch

# Import the base class - this is all we need for runtime checks
from rlaopt.linops.base import _BaseLinOp

# Type checking imports (only for static type checking)
if TYPE_CHECKING:
    from rlaopt.linops.simple import LinOp, TwoSidedLinOp, SymmetricLinOp
    from rlaopt.linops.distributed import (
        DistributedLinOp,
        DistributedTwoSidedLinOp,
        DistributedSymmetricLinOp,
    )

# Type annotation for static type checking
LinOpType = Union[
    "LinOp",
    "TwoSidedLinOp",
    "SymmetricLinOp",
    "DistributedLinOp",
    "DistributedTwoSidedLinOp",
    "DistributedSymmetricLinOp",
]


def _is_linop_or_torch_tensor(param: Any, param_name: str):
    # Use the base class for runtime checks - this is all we need
    if not isinstance(param, (_BaseLinOp, torch.Tensor)):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type LinOpType or torch.Tensor"
        )


__all__ = ["LinOpType", "_is_linop_or_torch_tensor"]
