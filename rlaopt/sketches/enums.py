from enum import Enum, auto


class _SketchSide(Enum):
    LEFT = auto()
    RIGHT = auto()

    @classmethod
    def _from_str(cls, value, param_name):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.lower()
            if value == "left":
                return cls.LEFT
            elif value == "right":
                return cls.RIGHT

        raise ValueError(
            f"Invalid value for {param_name}: {value}. "
            "Expected 'left', 'right', SketchSide.LEFT, "
            "or SketchSide.RIGHT."
        )
