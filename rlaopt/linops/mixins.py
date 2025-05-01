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

    def _apply_scaling(self, fn):
        """Create a scaled version of a callable function.

        Args:
            fn: A callable function whose results should be scaled.

        Returns:
            A function that applies scaling to the result of the original function.
            If scaling is 1.0, returns the original function unchanged.
        """
        if not hasattr(self, "_scaling") or self._scaling == 1.0:
            return fn  # No scaling needed

        # Create a callable object that wraps the function
        return _ScaledFunction(fn, self._scaling)
