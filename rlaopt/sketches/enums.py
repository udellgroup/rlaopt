from enum import Enum, auto


class _SketchSide(Enum):
    """Enumeration for sketching sides.

    Attributes:
        LEFT: Sketch from the left.
        RIGHT: Sketch from the right.
    """

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
            "Expected 'left', 'right', _SketchSide.LEFT, "
            "or _SketchSide.RIGHT."
        )


class _SketchMode(Enum):
    """Enumeration for sketching modes.

    Attributes:
        GAUSS: Gaussian sketching mode.
        ORTHO: Orthonormal sketching mode.
        SPARSE: Sparse sketching mode.
    """

    GAUSS = auto()
    ORTHO = auto()
    SPARSE = auto()

    @classmethod
    def _from_str(cls, value, param_name):
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value = value.lower()
            if value == "gauss":
                return cls.GAUSS
            elif value == "ortho":
                return cls.ORTHO
            elif value == "sparse":
                return cls.SPARSE

        raise ValueError(
            f"Invalid value for {param_name}: {value}. "
            "Expected 'gauss', 'ortho', 'sparse', "
            "_SketchMode.GAUSS, _SketchMode.ORTHO, or _SketchMode.SPARSE."
        )
