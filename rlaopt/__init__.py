"""rlaopt: Randomized linear algebra-based methods for optimization and scientific computing."""  # noqa: E501

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rlaopt")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
