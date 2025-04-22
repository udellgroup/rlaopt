from typing import Union
from dataclasses import dataclass, asdict

import torch
from rlaopt.utils import _is_float


__all__ = ["KernelConfig"]


@dataclass(kw_only=True, frozen=False)
class KernelConfig:
    const_scaling: float = 1.0
    lengthscale: Union[float, torch.Tensor]

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return asdict(self)

    def __post_init__(self):
        _is_float(self.const_scaling, "const_scaling")
        if not isinstance(self.lengthscale, (float, torch.Tensor)):
            raise TypeError(
                f"lengthscale is of type {type(self.lengthscale).__name__}, "
                "but expected type float or torch.Tensor"
            )
