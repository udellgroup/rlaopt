from typing import Any
from dataclasses import dataclass, asdict

import torch
from rlaopt.utils import _is_float


__all__ = ["KernelConfig"]


@dataclass(kw_only=True, frozen=False)
class KernelConfig:
    const_scaling: float = 1.0
    lengthscale: float | torch.Tensor

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return asdict(self)

    def to(self, device: torch.device) -> "KernelConfig":
        """Move the configuration to a specified device.

        Args:
            device (torch.device): The target device.

        Returns:
            KernelConfig: A new configuration object on the specified device.
        """
        if isinstance(self.lengthscale, torch.Tensor):
            return KernelConfig(
                const_scaling=self.const_scaling,
                lengthscale=self.lengthscale.to(device),
            )
        return self

    def __post_init__(self):
        """Post-initialization checks for the configuration.

        If the lengthscale is a 1D tensor, it is reshaped to a 2D tensor.

        Raises:
            TypeError: If the types of the parameters are incorrect.
            ValueError: If the shape of the lengthscale parameter is incorrect.
        """
        _is_float(self.const_scaling, "const_scaling")
        if not isinstance(self.lengthscale, (float, torch.Tensor)):
            raise TypeError(
                f"lengthscale is of type {type(self.lengthscale).__name__}, "
                "but expected type float or torch.Tensor"
            )
        if isinstance(self.lengthscale, torch.Tensor):
            if self.lengthscale.ndim != 1:
                raise ValueError(
                    f"lengthscale has {self.lengthscale.ndim} dimensions, "
                    "but expected 1 dimension"
                )


def _is_kernel_config(param: Any, param_name: str):
    if not isinstance(param, KernelConfig):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type KernelConfig"
        )
