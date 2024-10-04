from collections.abc import Sequence
from typing import Union


class SketchyOptsError(Exception):
    def __init__(self, message: str) -> None:
        error_page = "https://udellgroup.github.io/sketchyopts/api/errors.html"
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        error_msg = f"{message} (see {error_page}#{module_name}.{class_name})"
        super().__init__(error_msg)


class InputDimError(SketchyOptsError):
    r"""Incorrect Input Dimension.

    This error occurs when an argument of a function has unexpected dimension.
    """

    def __init__(
        self,
        input_name: str,
        actual_dim: int,
        required_dim: Union[int, Sequence[int]],
        custom_msg="",
    ) -> None:

        if custom_msg:
            msg = custom_msg
        elif isinstance(required_dim, Sequence) and len(required_dim) > 1:
            msg = f"Input {input_name} is expected to have any dimension in {required_dim} but has {actual_dim}."
        else:
            msg = f"Input {input_name} is expected to have dimension {required_dim} but has {actual_dim}."

        super().__init__(msg)


class MatrixNotSquareError(SketchyOptsError):
    r"""Non-square Matrix Error.

    This error occurs when an argument representing a non-square matrix gets passed to a
    function that expects a square matrix.
    """

    def __init__(self, input_name: str, actual_shape: tuple[int, ...]) -> None:
        super().__init__(
            f"Input {input_name} is expected to be a square matrix but has shape {actual_shape}."
        )
