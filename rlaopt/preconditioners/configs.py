from abc import ABC
from dataclasses import dataclass
import torch


@dataclass(kw_only=True, frozen=False)
class PreconditionerConfig(ABC):
    pass


@dataclass(kw_only=True, frozen=False)
class IdentityConfig(PreconditionerConfig):
    pass


@dataclass(kw_only=True, frozen=False)
class NewtonConfig(PreconditionerConfig):
    rho: float
    device: torch.device

    def __post_init__(self):
        # Check that "device" and "rho" are valid
        # TODO make these checks helper functions
        if not isinstance(self.rho, float) or self.rho < 0:
            raise ValueError("rho must be a non-negative float")
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device object")


@dataclass(kw_only=True, frozen=False)
class NystromConfig(PreconditionerConfig):
    rank: int
    rho: float
    device: torch.device
    type: str = "gauss"

    def __post_init__(self):
        # Check that "rank" and "rho" are valid
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positve integer")
        if not isinstance(self.rho, float) or self.rho < 0:
            raise ValueError("rho must be a non-negative float")
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device object")
