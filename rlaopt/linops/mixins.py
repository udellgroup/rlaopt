"""Mixins for extending linear operator functionality."""
from collections.abc import Callable


__all__ = ["ScaleMixin"]


class _ScaledFunction(Callable):
    """Callable class that wraps a function and applies scaling to its result."""

    def __init__(self, fn, scale):
        self.fn = fn
        self.scale = scale

        # Copy attributes from the original function
        if hasattr(fn, "__name__"):
            self.__name__ = f"scaled_{fn.__name__}"
        if hasattr(fn, "__doc__") and fn.__doc__:
            self.__doc__ = f"Scaled version of: {fn.__doc__}"

        # Copy other useful function attributes
        for attr in ["__module__", "__qualname__", "__annotations__", "__defaults__"]:
            if hasattr(fn, attr):
                setattr(self, attr, getattr(fn, attr))

    def __call__(self, *args, **kwargs):
        """Call the wrapped function and scale its result."""
        result = self.fn(*args, **kwargs)
        return self.scale * result


class ScaleMixin:
    """Mixin that adds scaling functionality to linear operators."""

    def _initialize_scaling(self, scale: float):
        """Initialize scaling factor.

        Args:
            scale: A floating-point scaling factor.
        """
        # Ensure scale is a float
        self._scaling = float(scale) if scale is not None else 1.0

    def _apply_scaling(self, obj):
        """Apply scaling to a result or callable function.

        This method can handle both direct results and callable functions:
        - If obj is callable: Returns a scaled version of the callable
        - If obj is not callable: Directly scales and returns the result

        Args:
            obj: Either a callable function whose results should be scaled,
                 or a direct result (like a tensor) to scale.

        Returns:
            If obj is callable: A function that applies scaling to the result.
            If obj is not callable: The scaled result.
            If scaling is 1.0, returns the original obj unchanged.
        """
        if not hasattr(self, "_scaling") or self._scaling == 1.0:
            return obj  # No scaling needed

        # For callable objects, return a scaled function
        if callable(obj):
            # For existing _ScaledFunction objects, combine the scales
            if isinstance(obj, _ScaledFunction):
                return _ScaledFunction(obj.fn, self._scaling * obj.scale)
            # For regular callables, create a new scaled function
            return _ScaledFunction(obj, self._scaling)

        # For direct results, apply scaling immediately
        return self._scaling * obj
