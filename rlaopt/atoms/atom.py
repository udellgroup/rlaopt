"""Base class for optimization atoms."""

from abc import ABC, abstractmethod
import torch
import cvxpy as cp
from typing import Optional


class Atom(torch.nn.Module, ABC):
    """Abstract base class for optimization atoms.

    An atom represents a mathematical function that can be used in optimization
    problems. Atoms have various properties (smooth, proxable, etc.) and can be composed
    to form more complex objective functions.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Evaluates the atom and returns its value as a tensor."""
        pass

    @abstractmethod
    def is_smooth(self) -> bool:
        """Returns True if the atom is smooth (differentiable everywhere)."""
        pass

    @abstractmethod
    def gradient(self) -> Optional[torch.Tensor]:
        """Returns the gradient of the atom.

        This method should only be called if the atom is smooth. Otherwise, it should
        raise a NotImplementedError.
        """
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
        """
        pass

    @abstractmethod
    def to_cvxpy(self) -> cp.Expression:
        """Converts the atom to a CVXPY expression for convex optimization.

        Returns:
            CVXPY expression representing this atom
        """
        pass

    @abstractmethod
    def __mul__(self, scalar: float) -> "Atom":
        """Scale the atom by a scalar.

        Each atom handles its own scaling.
        """
        pass

    def __rmul__(self, scalar: float) -> "Atom":
        """Scale the atom by a scalar (reverse multiplication)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Atom":
        """Divide the atom by a scalar."""
        return self.__mul__(1.0 / scalar)

    def __add__(self, other: "Atom") -> "SumAtom":
        """Add atoms together."""
        if isinstance(other, Atom):
            return SumAtom([self, other])
        return NotImplemented

    def __radd__(self, other: "Atom") -> "SumAtom":
        """Add atoms together (reverse addition)."""
        return self.__add__(other)

    def __sub__(self, other: "Atom") -> "SumAtom":
        """Subtract atoms (self - other)."""
        if isinstance(other, Atom):
            return SumAtom([self, other * (-1.0)])
        return NotImplemented

    def __rsub__(self, other: "Atom") -> "SumAtom":
        """Reverse subtraction (other - self)."""
        if isinstance(other, Atom):
            return other.__sub__(self)
        return NotImplemented


class SumAtom(Atom):
    """Sum of multiple atoms."""

    def __init__(self, atoms: list[Atom]):
        super().__init__()

        # Flatten nested SumAtoms during initialization
        flattened_atoms = []
        for atom in atoms:
            if isinstance(atom, SumAtom):
                flattened_atoms.extend(list(atom.atoms))
            else:
                flattened_atoms.append(atom)

        self.atoms = torch.nn.ModuleList(flattened_atoms)

    def forward(self) -> torch.Tensor:
        return sum(atom.forward() for atom in self.atoms)

    def is_smooth(self) -> bool:
        return all(atom.is_smooth() for atom in self.atoms)

    def gradient(self) -> torch.Tensor:
        return sum(atom.gradient() for atom in self.atoms)

    def is_proxable(self) -> bool:
        # Default to False - subclasses should override with specific logic
        return False

    def prox(self, location: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SumAtom does not have a prox operator by default.")

    def is_subsamplable(self) -> bool:
        # Default to False - subclasses should override with specific logic
        return False

    def subsample(self, indices: torch.Tensor) -> "SumAtom":
        raise NotImplementedError("SumAtom does not support subsampling by default.")

    def __mul__(self, scalar: float) -> "SumAtom":
        """Scale all atoms in the sum."""
        return SumAtom([atom * scalar for atom in self.atoms])

    def to_cvxpy(self) -> cp.Expression:
        return sum(atom.to_cvxpy() for atom in self.atoms)
