from enum import Enum, auto


class _Operation(Enum):
    MATVEC = auto()
    RMATVEC = auto()


class _DistributionMode(Enum):
    ROW = auto()  # Matrix is distributed across rows
    COLUMN = auto()  # Matrix is distributed across columns

    @classmethod
    def _from_str(cls, value, param_name):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.lower()
            if value == "row":
                return cls.ROW
            elif value == "column":
                return cls.COLUMN

        raise ValueError(
            f"Invalid value for {param_name}: {value}. "
            "Expected 'row', 'column', _DistributionMode.ROW, "
            "or _DistributionMode.COLUMN."
        )
