from enum import Enum, auto

__all__ = ["DistributionMode"]


class _Operation(Enum):
    MATVEC = auto()
    RMATVEC = auto()


class DistributionMode(Enum):
    ROW = auto()  # Matrix is distributed across rows
    COLUMN = auto()  # Matrix is distributed across columns

    @classmethod
    def _from_str(cls, value):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.lower()
            if value == "row":
                return cls.ROW
            elif value == "column":
                return cls.COLUMN

        raise ValueError(
            f"Invalid distribution mode: {value}. "
            "Expected 'row', 'column', DistributionMode.ROW, "
            "or DistributionMode.COLUMN."
        )
