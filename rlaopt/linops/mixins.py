"""Mixins for extending linear operator functionality."""

__all__ = ["ScaleMixin"]


class ScaleMixin:
    """Mixin that adds scaling functionality to linear operators."""

    def _initialize_scaling(self, scale: float):
        """Initialize scaling factor.

        Args:
            scale: A floating-point scaling factor.
        """
        # Ensure scale is a float
        self._scaling = float(scale) if scale is not None else 1.0

    def _apply_scaling(self, result):
        """Apply scaling to a result.

        Args:
            result: The result to scale.

        Returns:
            The scaled result if scaling is not 1.0, otherwise the original result.
        """
        if hasattr(self, "_scaling") and self._scaling != 1.0:
            return self._scaling * result
        return result

    def _scale_linop(self, linop):
        """Apply scaling to a linear operator.

        Args:
            linop: The linear operator to scale.

        Returns:
            A ScaleLinOp wrapping the input operator if scaling is not 1.0,
            otherwise the original operator.
        """
        if hasattr(self, "_scaling") and self._scaling != 1.0:
            from .scale import ScaleLinOp

            return ScaleLinOp(linop, self._scaling)
        return linop
