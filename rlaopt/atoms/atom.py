"""Base class for optimization atoms."""

from abc import ABC, abstractmethod

import torch

from rlaopt.expression import Expression


class Atom(Expression, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, etc.) and can be composed
    to form more complex objective functions.
    """

    def __init__(self):
        """Initializes the atom.

        Subclasses should call this constructor to ensure proper initialization.
        Subclasses should also register any variables they use with the atom.
        """
        super().__init__()

    @abstractmethod
    def evaluate_at(self, **variable_locations):
        """Evaluate the atom at specific variable locations.

        Args:
            **variable_locations: Mapping of variable names to locations

        Returns:
            Value of the atom with variables substituted with their locations
        """
        pass

    @abstractmethod
    def is_proxable(self) -> bool:
        """Returns True if the atom has a computable proximal operator."""
        pass

    @abstractmethod
    def prox(self, location: torch.Tensor, prox_scaling: float) -> torch.Tensor:
        """Proximal operator of the atom.

        This method should only be called if the atom is proxable. Otherwise, it should
        raise a NotImplementedError.

        Args:
            location: Point at which to evaluate the proximal operator
            prox_scaling: Scaling factor for the proximal operator

        Returns:
            Result of the proximal operator
        """
        pass

    @abstractmethod
    def is_subsamplable(self) -> bool:
        """Returns True if the atom supports subsampling (e.g., for stochastic
        methods)."""
        pass

    @abstractmethod
    def subsample(self, indices: torch.Tensor) -> "Atom":
        """Returns a subsampled version of the atom.

        This method should only be called if the atom is subsamplable.
        Otherwise, it should raise a NotImplementedError.

        Args:
            indices: Indices to subsample

        Returns:
            New atom representing the subsampled version

        Raises:
            NotImplementedError: If the atom does not support subsampling.
        """
        pass
