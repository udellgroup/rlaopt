from enum import Enum, auto


class _DampingMode(Enum):
    """Enumeration for different damping modes.

    Attributes:
        ADAPTIVE: Adaptive damping mode.
        NON_ADAPTIVE: Fixed damping mode.
    """

    ADAPTIVE = auto()
    NON_ADAPTIVE = auto()

    @classmethod
    def _from_str(cls, value, param_name):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.lower()
            if value == "adaptive":
                return cls.ADAPTIVE
            elif value == "non_adaptive":
                return cls.NON_ADAPTIVE

        raise ValueError(
            f"Invalid value for {param_name}: {value}. "
            "Expected 'adaptive', 'non_adaptive', DampingMode.ADAPTIVE, "
            "or DampingMode.NON_ADAPTIVE."
        )
