"""Base class for optimization atoms."""

from abc import ABC, abstractmethod
import torch
import cvxpy as cp


class Atom(torch.nn.Module, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, etc.) and can be composed
    to form more complex objective functions.
    """

    def __init__(self):
        """Initializes the atom."""
        super().__init__()

    @abstractmethod
    def forward(self, location: torch.Tensor) -> torch.Tensor:
        """Evaluates the atom and returns its value as a tensor.

        Args:
            location: Point at which to evaluate the atom

        Returns:
            Value of the atom at the specified location
        """
        pass

    @abstractmethod
    def is_smooth(self) -> bool:
        """Returns True if the atom is smooth (differentiable everywhere)."""
        pass

    @abstractmethod
    def is_proxable(self) -> bool:
        """Returns True if the atom has a computable proximal operator."""
        pass

    @abstractmethod
    def prox(self, location: torch.Tensor) -> torch.Tensor:
        """Proximal operator of the atom.

        This method should only be called if the atom is proxable. Otherwise, it should
        raise a NotImplementedError.

        Note that this function has to account for the scaling factor of the atom.

        Args:
            location: Point at which to evaluate the proximal operator

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

    @abstractmethod
    def to_cvxpy(self, expr: cp.Expression) -> cp.Expression:
        """Converts the atom to a CVXPY expression.

        This method should be called with a CVXPY expression, including
        its subclasses of CVXPY variables and CVXPY parameters.

        Args:
            expr: Either a cp.Variable or a cp.Expression

        Returns:
            CVXPY expression representing this atom
        """
        pass
