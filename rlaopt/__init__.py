"""rlaopt: randomized linear algebra tools for large-scale optimization."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlaopt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
