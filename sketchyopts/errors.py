class SketchyOptsError(Exception):
    def __init__(self, message):
        error_page = "https://udellgroup.github.io/sketchyopts/api/errors.html"
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        error_msg = f"{message} (see {error_page}#{module_name}.{class_name})"
        super().__init__(error_msg)


class InputDimError(SketchyOptsError):

    def __init__(self, input_name, actual_dim, required_dim):
        super().__init__(
            f"Input {input_name} is expected to have dimension {required_dim} but has {actual_dim}."
        )


class MatrixNotSquareError(SketchyOptsError):

    def __init__(self, input_name, actual_shape):
        super().__init__(
            f"Input {input_name} is expected to be a square matrix but has shape {actual_shape}."
        )
