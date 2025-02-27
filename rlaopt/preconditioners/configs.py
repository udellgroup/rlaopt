from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any

from rlaopt.utils import _is_str, _is_nonneg_float, _is_pos_int


@dataclass(kw_only=True, frozen=False)
class PreconditionerConfig(ABC):
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True, frozen=False)
class IdentityConfig(PreconditionerConfig):
    pass


@dataclass(kw_only=True, frozen=False)
class NewtonConfig(PreconditionerConfig):
    rho: float

    def __post_init__(self):
        _is_nonneg_float(self.rho, "rho")


@dataclass(kw_only=True, frozen=False)
class NystromConfig(PreconditionerConfig):
    rank: int
    rho: float
    sketch: str = "gauss"

    def __post_init__(self):
        _is_pos_int(self.rank, "rank")
        _is_nonneg_float(self.rho, "rho")
        _is_str(self.sketch, "sketch")


def _is_precond_config(param: Any, param_name: str):
    if not isinstance(param, PreconditionerConfig):
        raise TypeError(
            f"{param_name} is of type {type(param).__name__}, "
            "but expected type PreconditionerConfig"
        )
